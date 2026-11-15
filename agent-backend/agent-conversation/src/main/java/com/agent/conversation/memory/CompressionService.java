package com.agent.conversation.memory;

import com.agent.conversation.service.MessageCacheService;
import com.agent.conversation.service.impl.MessageInfo;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.*;

@Component
public class CompressionService {
    private static final Logger log = LoggerFactory.getLogger(CompressionService.class);
    private static final ObjectMapper mapper = new ObjectMapper();

    @Value("${ai.engine.url:http://localhost:8000}")
    private String engineUrl;

    private final MessageCacheService cacheService;
    private final com.agent.conversation.mapper.ConversationMapper conversationMapper;

    public CompressionService(MessageCacheService cacheService,
                              com.agent.conversation.mapper.ConversationMapper conversationMapper) {
        this.cacheService = cacheService;
        this.conversationMapper = conversationMapper;
    }

    /**
     * 压缩旧消息为摘要：保留最近 recentCount 条（recentRounds 轮）原文不动，更早的合并进摘要。
     * 压缩成功后把已压缩消息从 Redis 窗口弹出（只留最近 recentCount 条）——窗口本身即
     * "未压缩队列"，同一批消息不会被二次压缩；压缩失败则整窗原样保留，下一轮自动重试。
     * 触发时机由 token 预算决定，弹出条数不固定（= 窗口大小 - recentCount）。
     * @return { summary, recentMessages }
     */
    public CompressionResult buildCompressedContext(List<MessageInfo> allMessages, int recentRounds,
                                                     Long conversationId) {
        int recentCount = recentRounds * 2;
        if (allMessages.size() <= recentCount) {
            return new CompressionResult(null, allMessages);
        }

        List<MessageInfo> older = allMessages.subList(0, allMessages.size() - recentCount);
        List<MessageInfo> recent = allMessages.subList(allMessages.size() - recentCount, allMessages.size());

        String existingSummary = cacheService.getSummary(conversationId);
        String newSummary = compress(older, existingSummary);
        if (newSummary == null) {
            // 压缩失败：不写摘要、不弹出窗口，本轮用原文顶着，等 API 恢复后下轮重试
            log.warn("Compression failed, keeping {} messages for conv {}", allMessages.size(), conversationId);
            return new CompressionResult(existingSummary, allMessages);
        }

        // 持久化到 PostgreSQL（源头），再写 Redis 缓存，最后弹出已压缩消息
        var conv = conversationMapper.selectById(conversationId);
        if (conv != null) {
            conv.setSummary(newSummary);
            conversationMapper.updateById(conv);
        }
        cacheService.setSummary(conversationId, newSummary);
        cacheService.trimHistory(conversationId, recentCount);
        log.info("Compressed {} older messages into summary ({} chars), trimmed window to {} for conv {}",
            older.size(), newSummary.length(), recentCount, conversationId);

        return new CompressionResult(newSummary, recent);
    }

    /** 调用引擎 /memory/summarize 压缩旧消息并拼接旧摘要；失败返回 null（由调用方决定降级行为） */
    public String compress(List<MessageInfo> olderMessages, String existingSummary) {
        try {
            URI uri = URI.create(engineUrl + "/memory/summarize");
            HttpURLConnection conn = (HttpURLConnection) uri.toURL().openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(60000);

            List<Map<String, String>> msgs = new ArrayList<>();
            for (MessageInfo m : olderMessages) {
                msgs.add(Map.of("role", m.role(), "content", m.content()));
            }

            Map<String, Object> body = new HashMap<>();
            body.put("messages", msgs);

            String json = mapper.writeValueAsString(body);
            try (var os = conn.getOutputStream()) {
                os.write(json.getBytes(StandardCharsets.UTF_8));
            }

            if (conn.getResponseCode() != 200) {
                log.warn("Summarize API returned {}", conn.getResponseCode());
                return null;
            }

            var response = mapper.readValue(conn.getInputStream(), Map.class);
            String summary = (String) response.get("summary");
            if (summary == null || summary.isBlank()) {
                // 空摘要按失败处理：避免把消息压成空白后弹出导致信息丢失
                return null;
            }
            if (existingSummary != null && !existingSummary.isEmpty()) {
                summary = existingSummary + "\n" + summary;
            }
            return summary;
        } catch (IOException e) {
            log.error("Summarize API failed: {}", e.getMessage());
            return null;
        }
    }

    public record CompressionResult(String summary, List<MessageInfo> recentMessages) {}
}

package com.agent.knowledge.service.impl;

import com.agent.knowledge.service.DocIngestService;

import com.agent.knowledge.chunker.ImageChunker;
import com.agent.knowledge.chunker.TextChunker;
import com.agent.knowledge.entity.Chunk;
import com.agent.common.BusinessException;
import com.agent.common.ErrorCode;
import com.agent.knowledge.entity.Document;
import com.agent.knowledge.parser.DocumentParser;
import com.agent.knowledge.mapper.ChunkMapper;
import com.agent.knowledge.mapper.DocumentMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class DocIngestServiceImpl implements DocIngestService {

    private static final Logger log = LoggerFactory.getLogger(DocIngestServiceImpl.class);

    /** 与 agent-gateway 的 MqConfig 保持一致（knowledge 模块不依赖 gateway，故在此声明） */
    private static final String EXCHANGE_AGENT = "agent.exchange";
    private static final String ROUTING_DOC_PROCESS = "doc.process";
    /** Celery 任务名（见 agent-engine tasks/doc_process.py） */
    private static final String TASK_DOC_PROCESS = "doc.process";

    @Autowired
    private List<DocumentParser> parsers;

    @Autowired
    private RabbitTemplate rabbitTemplate;

    @Autowired
    private TextChunker textChunker;

    @Autowired
    private ImageChunker imageChunker;

    @Autowired
    private DocumentMapper documentMapper;

    @Autowired
    private ChunkMapper chunkMapper;

    @Override
    @Transactional
    public Document ingest(MultipartFile file, Long tenantId) throws Exception {
        String fileName = file.getOriginalFilename();
        String ext = getFileExtension(fileName);

        Document doc = new Document();
        doc.setFileName(fileName);
        doc.setFileType(ext);
        doc.setFileSize(file.getSize());
        doc.setTenantId(tenantId);
        doc.setStatus("PENDING");
        documentMapper.insert(doc);

        try {
            doc.setStatus("PARSING");
            documentMapper.updateById(doc);

            DocumentParser parser = findParser(ext);
            if (parser == null) {
                throw new BusinessException(ErrorCode.FILE_TYPE_UNSUPPORTED, "不支持的文件类型: " + ext);
            }
            Map<String, String> parsed = parser.parse(file.getInputStream(), fileName);

            doc.setStatus("CHUNKING");
            documentMapper.updateById(doc);

            String text = parsed.get("text");
            String images = parsed.get("images");
            boolean needsVision = "true".equals(parsed.get("needsVision"));

            int chunkIndex = 0;
            List<Chunk> pendingChunks = new ArrayList<>();

            if (text != null && !text.isBlank()) {
                List<String> textChunks = textChunker.chunk(text);
                for (String chunkContent : textChunks) {
                    Chunk chunk = new Chunk();
                    chunk.setDocumentId(doc.getId());
                    chunk.setContent(chunkContent);
                    chunk.setChunkType("text");
                    chunk.setChunkIndex(chunkIndex++);
                    chunkMapper.insert(chunk);
                    pendingChunks.add(chunk);
                }
            }

            if (images != null && !images.isBlank()) {
                String[] imagePathArray = images.split(";");
                for (String imgPath : imagePathArray) {
                    if (imgPath.isBlank()) continue;
                    Chunk chunk = new Chunk();
                    chunk.setDocumentId(doc.getId());
                    chunk.setContent(needsVision ? "[需要多模态处理]" : "");
                    chunk.setChunkType("image");
                    chunk.setImagePath(imgPath.trim());
                    chunk.setChunkIndex(chunkIndex);
                    chunkMapper.insert(chunk);
                    chunkIndex++;
                    pendingChunks.add(chunk);
                }
            }

            doc.setChunkCount(chunkIndex);
            doc.setStatus("DONE");
            documentMapper.updateById(doc);

            // 事务提交后投递向量化消息（Celery 异步消费）
            sendVectorizeMessagesAfterCommit(pendingChunks);

            log.info("文档摄取完成: {}, 分片数: {}", fileName, chunkIndex);
            return doc;

        } catch (Exception e) {
            log.error("文档摄取失败: {}", fileName, e);
            doc.setStatus("FAILED");
            doc.setErrorMsg(e.getMessage());
            documentMapper.updateById(doc);
            throw e;
        }
    }

    private DocumentParser findParser(String ext) {
        if (ext == null) return null;
        String type = ext.toLowerCase();

        if (isImageType(type)) return findParserByType("image");

        return findParserByType(type);
    }

    private DocumentParser findParserByType(String type) {
        return parsers.stream()
                .filter(p -> p.supportedType().equalsIgnoreCase(type))
                .findFirst()
                .orElse(null);
    }

    private boolean isImageType(String ext) {
        for (String imgExt : com.agent.knowledge.parser.ImageParser.SUPPORTED_EXTENSIONS) {
            if (imgExt.equalsIgnoreCase(ext)) return true;
        }
        return false;
    }

    /**
     * 事务提交后才投递向量化消息（Celery 消费）。
     * 若在事务内直接发送，消费者可能在事务提交前处理消息：
     * 回写 vectorized 标记会落空，事务回滚后还会产生指向不存在 chunk 的"幻影消息"。
     */
    private void sendVectorizeMessagesAfterCommit(List<Chunk> pendingChunks) {
        if (pendingChunks == null || pendingChunks.isEmpty()) return;
        TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
            @Override
            public void afterCommit() {
                for (Chunk chunk : pendingChunks) {
                    try {
                        sendVectorizeMessage(chunk);
                    } catch (Exception e) {
                        // 发送失败不影响主流程：chunk 仍为 vectorized=false，可由补偿任务重发
                        log.error("投递向量化消息失败, chunkId={}", chunk.getId(), e);
                    }
                }
            }
        });
    }

    /** 按 Celery protocol v2 组装消息体: [args, kwargs, embed]，并由 header.task 路由到任务函数 */
    private void sendVectorizeMessage(Chunk chunk) {
        List<Object> args = List.of(
                chunk.getId(),
                chunk.getContent() != null ? chunk.getContent() : "",
                chunk.getChunkType());

        Map<String, Object> kwargs = new HashMap<>();
        if ("image".equals(chunk.getChunkType()) && chunk.getImagePath() != null) {
            kwargs.put("image_path", chunk.getImagePath());
        }
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("documentId", chunk.getDocumentId());
        metadata.put("chunkIndex", chunk.getChunkIndex());
        kwargs.put("metadata", metadata);

        List<Object> body = List.of(args, kwargs, Map.of());

        rabbitTemplate.convertAndSend(EXCHANGE_AGENT, ROUTING_DOC_PROCESS, body, message -> {
            message.getMessageProperties().setHeader("task", TASK_DOC_PROCESS);
            message.getMessageProperties().setHeader("id", UUID.randomUUID().toString());
            return message;
        });
    }

    private String getFileExtension(String fileName) {
        if (fileName == null || !fileName.contains(".")) return "";
        return fileName.substring(fileName.lastIndexOf('.') + 1);
    }
}

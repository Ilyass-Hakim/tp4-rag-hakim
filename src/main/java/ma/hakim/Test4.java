package ma.hakim;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import ma.hakim.assistant.Assistant;

import java.util.List;
import java.util.Scanner;

public class Test4 {

    public static void main(String[] args) {
        String cle = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash-lite")
                .logRequestsAndResponses(true)
                .build();

        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(1000, 100);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.5)
                .build();

        String cleTavily = System.getenv("TAVILY_KEY");
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(cleTavily)
                .build();

        ContentRetriever webSearchContentRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();

        QueryRouter queryRouter = new DefaultQueryRouter(contentRetriever, webSearchContentRetriever);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        conversationAvec(assistant);
    }

  private static void conversationAvec(Assistant assistant) {
    try (Scanner scanner = new Scanner(System.in)) {
      while (true) {
        System.out.println("==================================================");
        System.out.println("Posez votre question : ");
        String question = scanner.nextLine();
        if (question.isBlank()) {
          continue;
        }
        System.out.println("==================================================");
        if ("fin".equalsIgnoreCase(question)) {
          break;
        }
        String reponse = assistant.chat(question);
        System.out.println("Assistant : " + reponse);
        System.out.println("==================================================");
      }
    }
  }
}

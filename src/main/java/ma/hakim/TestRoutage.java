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
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.hakim.assistant.Assistant;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TestRoutage {

    public static void main(String[] args) {
        String cle = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash-lite")
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStoreRag = creerEmbeddingStore("rag.pdf", embeddingModel);
        EmbeddingStore<TextSegment> embeddingStoreDevops =
                creerEmbeddingStore("OCI_DevOps_Professional_Complete_Course.pdf", embeddingModel);

        ContentRetriever contentRetrieverRag = creerContentRetriever(embeddingStoreRag, embeddingModel);
        ContentRetriever contentRetrieverDevops = creerContentRetriever(embeddingStoreDevops, embeddingModel);

        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(contentRetrieverRag,
                "Cours sur l'intelligence artificielle : le fine-tuning et le RAG (Retrieval Augmented Generation) des modèles de langage");
        descriptions.put(contentRetrieverDevops,
                "Cours sur le DevOps avec Oracle Cloud Infrastructure (OCI) : intégration continue, déploiement continu et services du cloud Oracle");

        QueryRouter queryRouter = new LanguageModelQueryRouter(model, descriptions);

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

    private static EmbeddingStore<TextSegment> creerEmbeddingStore(String nomFichier, EmbeddingModel embeddingModel) {
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(nomFichier, documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(1000, 100);
        List<TextSegment> segments = splitter.split(document);

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }

    private static ContentRetriever creerContentRetriever(EmbeddingStore<TextSegment> embeddingStore,
                                                          EmbeddingModel embeddingModel) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.5)
                .build();
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

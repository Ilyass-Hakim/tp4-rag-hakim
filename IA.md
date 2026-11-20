# Utilisation de l'IA pour le TP 4

**IA utilisée :** Claude AI 

------------------------------------------------------------------------

## 1. Comprendre le fonctionnement de la phase d'ingestion

**Problème rencontré :**

Je devais comprendre les différentes étapes de la phase d'ingestion du
RAG, notamment le chargement du document, son découpage en segments et
la génération des embeddings. Le rôle de certains composants de
LangChain4j, comme `DocumentSplitter`, `TextSegment` et
`EmbeddingModel`, n'était pas encore totalement clair.

**Raison de l'utilisation de l'IA :**

J'ai utilisé l'IA pour obtenir une explication plus détaillée du
fonctionnement de ces composants et comprendre comment ils travaillent
ensemble avant de passer à l'implémentation.

**Prompt utilisé :**

> Je comprends le principe général de l'ingestion dans un système RAG,
> mais j'aimerais mieux comprendre le rôle de DocumentSplitter,
> TextSegment et EmbeddingModel dans LangChain4j. Explique-moi comment
> ces composants travaillent ensemble, avec un petit exemple concret, et
> pourquoi il est nécessaire de découper un document avant de générer
> ses embeddings.

**Réponse de l'IA (extrait) :**

La phase d'ingestion prépare les documents pour permettre leur recherche
ultérieure. Le `DocumentSplitter` découpe le document en morceaux plus
petits. Chaque morceau devient un `TextSegment`. Ensuite,
l'`EmbeddingModel` transforme le texte de chaque segment en un vecteur
numérique appelé *embedding*. Enfin, le `EmbeddingStore` conserve les
embeddings avec les segments correspondants.

Par exemple, au lieu de créer un seul embedding pour un PDF de plusieurs
dizaines de pages, on crée un embedding pour chaque segment. Cela permet
de retrouver uniquement les passages qui correspondent le mieux à une
question, plutôt que de récupérer tout le document.

**Ce que j'ai appris :**

J'ai compris que la phase d'ingestion consiste à préparer les documents
avant les recherches. Le document est d'abord découpé en plusieurs
segments (`TextSegment`), puis chaque segment est transformé en vecteur
numérique appelé embedding. Enfin, les embeddings et les segments
associés sont enregistrés dans un `EmbeddingStore` afin de pouvoir
retrouver plus tard les passages pertinents.

------------------------------------------------------------------------

## 2. Comprendre et interpréter les logs

**Problème rencontré :**

Après avoir activé le logging de LangChain4j, les logs générés pendant
une requête RAG peuvent être difficiles à interpréter. Il fallait
notamment comprendre quelles informations sont récupérées et comment
elles sont ajoutées à la question envoyée au modèle Gemini.

**Raison de l'utilisation de l'IA :**

J'ai utilisé l'IA pour m'aider à lire les logs et identifier les
différentes étapes de l'échange entre mon application, le système de
récupération et le modèle de langage.

**Prompt utilisé :**

> Je viens d'activer le logging dans mon application LangChain4j.
> Explique-moi comment interpréter les logs générés pendant une requête
> RAG, notamment comment identifier la question originale, les segments
> récupérés et le prompt final envoyé au modèle Gemini.

**Réponse de l'IA (extrait) :**

Dans les logs de mon projet, on ne voit pas une ligne séparée qui décrit la recherche effectuée dans l'`EmbeddingStore`. On voit surtout les requêtes envoyées à Gemini. Les segments récupérés sont visibles directement dans le contenu de la requête envoyée au modèle, après la partie « Answer using the following information: ». Cette partie permet donc d'identifier le contexte documentaire ajouté à la question de l'utilisateur avant son envoi à Gemini.

**Ce que j'ai appris :**

J'ai appris à repérer dans les logs la requête envoyée à Gemini et à reconnaître les segments documentaires ajoutés après « Answer using the following information: ». Dans mon projet, les logs ne montrent pas séparément l'opération de recherche dans l'`EmbeddingStore` ; ils permettent surtout d'observer le contexte récupéré à travers le prompt transmis au modèle.

------------------------------------------------------------------------

## 3. Comprendre le problème et le QueryTransformer

**Problème rencontré :**

Lors d'une conversation, une deuxième question peut dépendre de la
première. Par exemple, après avoir demandé « Qu'est-ce que le
fine-tuning ? », l'utilisateur peut demander « Dans le support de cours,
à quoi c'est comparé ? ». Cette deuxième question ne contient pas
suffisamment de contexte lorsqu'elle est utilisée seule pour effectuer
une recherche.

**Raison de l'utilisation de l'IA :**

J'ai utilisé l'IA pour comprendre pourquoi ce type de question peut
poser problème au système RAG et découvrir comment un `QueryTransformer`
peut reformuler la question en tenant compte du contexte de la
conversation.

**Prompt utilisé :**

> Dans mon application RAG, si je pose une première question comme «
> Qu'est-ce que le fine-tuning ? », puis une deuxième question comme «
> Dans le support de cours, à quoi c'est comparé ? », le système risque
> de ne pas retrouver les bons passages. Explique-moi pourquoi ce
> problème se produit et comment un QueryTransformer permet de le
> résoudre dans LangChain4j. Donne-moi un exemple concret.

**Réponse de l'IA (extrait) :**

Le problème vient du fait que la deuxième question est incomplète
lorsqu'elle est considérée séparément. Le système de recherche peut ne
pas savoir à quoi se rapporte l'expression « c'est ». Il risque donc de
rechercher des passages qui ne correspondent pas au sujet de la
conversation.

Un `QueryTransformer` peut reformuler cette question en utilisant
l'historique. Par exemple, il peut transformer « Dans le support de
cours, à quoi c'est comparé ? » en une question plus complète comme «
Dans le support de cours sur le RAG, à quoi le fine-tuning est-il
comparé ? ». Cette nouvelle formulation donne au mécanisme de recherche
davantage de contexte.

**Ce que j'ai appris :**

J'ai compris qu'une question isolée peut être ambiguë lorsqu'elle fait
référence à un élément évoqué précédemment. Le `QueryTransformer` permet
de transformer cette question en une requête plus explicite, en
utilisant le contexte de la conversation, afin d'améliorer la recherche
des passages pertinents.

------------------------------------------------------------------------

## 4. Choisir et implémenter le bon QueryTransformer

**Problème rencontré :**

Après avoir compris le problème des questions dépendantes du contexte,
il fallait identifier le `QueryTransformer` adapté et savoir comment
l'intégrer dans l'application existante sans modifier inutilement les
autres parties du code.

**Raison de l'utilisation de l'IA :**

J'ai utilisé l'IA pour identifier le composant adapté dans LangChain4j,
comprendre son fonctionnement et obtenir un exemple d'intégration
correspondant à l'architecture du TP.

**Prompt utilisé :**

> Je dois améliorer mon application RAG en utilisant un QueryTransformer
> pour prendre en compte le contexte de la conversation précédente. Quel
> QueryTransformer de LangChain4j est adapté à ce besoin ? Explique-moi
> son fonctionnement et montre-moi comment l'intégrer à mon application
> existante sans modifier inutilement le reste du code.

**Réponse de l'IA (extrait) :**

Pour ce besoin, le `CompressingQueryTransformer` est adapté : il utilise
le contexte de la conversation pour transformer une question de suivi en
une question autonome, plus facile à traiter par le système de
recherche. Il s'appuie sur un modèle de langage pour effectuer cette
reformulation.

Il faut ensuite intégrer ce transformer dans le processus de
récupération, au niveau du `RetrievalAugmentor`, plutôt que de modifier
manuellement chaque question dans le code de l'assistant. Il faut
vérifier les constructeurs et les méthodes disponibles dans la version
de LangChain4j utilisée par le projet, car l'API peut évoluer.

**Ce que j'ai appris :**

J'ai appris qu'il faut choisir le type de transformation en fonction du
problème rencontré. Dans ce cas, l'objectif est de reformuler une
question en tenant compte de l'historique de la conversation, avant de
lancer la recherche dans les documents.

------------------------------------------------------------------------

## 5. Comprendre le routage conditionnel

**Problème rencontré :**

Dans un système RAG, toutes les questions ne nécessitent pas forcément
une recherche dans les documents. Par exemple, une simple salutation
comme « Bonjour » ne nécessite pas de consulter le document contenant le
cours sur l'IA. Il fallait comprendre comment le système pouvait décider
de ne pas utiliser le RAG.

**Raison de l'utilisation de l'IA :**

J'ai utilisé l'IA pour comprendre le rôle d'un `QueryRouter`
personnalisé et le fonctionnement du retour d'une liste vide
lorsqu'aucune recherche documentaire n'est nécessaire.

**Prompt utilisé :**

> Je souhaite comprendre comment créer un système RAG capable de décider
> s'il doit utiliser la recherche documentaire ou répondre directement à
> l'utilisateur. Dans LangChain4j, comment implémenter une classe
> personnalisée qui respecte l'interface QueryRouter et qui utilise
> ChatModel pour déterminer si une question concerne l'IA ? Explique-moi
> particulièrement pourquoi le retour d'une liste vide permet de
> désactiver le RAG.

**Réponse de l'IA (extrait) :**

Le `QueryRouter` sert à choisir les `ContentRetriever` qui doivent être
utilisés pour traiter une question. Dans ce test, on peut créer une
implémentation personnalisée de cette interface et utiliser directement
`ChatModel` pour demander au modèle si la question concerne le domaine
couvert par le document.

Par exemple, on lui demande de répondre uniquement par « oui », « non »
ou « peut-être ». Si la réponse est « non », le routeur retourne une
liste vide : aucun `ContentRetriever` n'est alors sélectionné et la
recherche documentaire n'est pas effectuée. Si la réponse est « oui » ou
« peut-être », le routeur retourne le `ContentRetriever` du document.

**Ce que j'ai appris :**

J'ai compris qu'un `QueryRouter` personnalisé peut analyser la question
avant de choisir les sources à consulter. S'il ne retourne aucun
`ContentRetriever`, aucune recherche documentaire n'est effectuée pour
cette requête. Cela permet d'éviter d'ajouter des extraits sans rapport
avec la question.

------------------------------------------------------------------------

## 6. Comprendre l'intégration de Tavily

**Problème rencontré :**

Le dernier test demande d'ajouter une recherche sur le Web au système
RAG existant. Il fallait comprendre comment intégrer Tavily et comment
associer les résultats Web aux informations récupérées dans le document
PDF.

**Raison de l'utilisation de l'IA :**

J'ai utilisé l'IA pour comprendre le rôle des composants de recherche
Web de LangChain4j et leur association avec les composants de
récupération déjà utilisés dans le TP.

**Prompt utilisé :**

> Je dois intégrer Tavily comme moteur de recherche Web dans mon
> application RAG avec LangChain4j. Explique-moi le rôle de
> WebSearchEngine et WebSearchContentRetriever, puis montre-moi comment
> les associer au ContentRetriever de mon document PDF avec
> DefaultQueryRouter et RetrievalAugmentor.

**Réponse de l'IA (extrait) :**

On dispose de deux récupérateurs : un `EmbeddingStoreContentRetriever` pour rechercher dans le PDF et un `WebSearchContentRetriever` pour effectuer une recherche sur le Web. Dans cette configuration, le `DefaultQueryRouter` renvoie toujours les deux `ContentRetriever` : la recherche est donc effectuée à la fois dans le PDF et sur le Web, quelle que soit la question posée. Le `RetrievalAugmentor` récupère ensuite les contenus des deux sources et les ajoute au contexte transmis au modèle.

**Ce que j'ai appris :**

J'ai compris que Tavily permet d'ajouter une source d'informations
externe au système RAG. `WebSearchEngine` représente le moteur de
recherche Web et `WebSearchContentRetriever` permet d'utiliser ses
résultats comme contenu récupérable. Avec un `DefaultQueryRouter`,
plusieurs sources peuvent être utilisées pour enrichir la question
envoyée au modèle.

------------------------------------------------------------------------

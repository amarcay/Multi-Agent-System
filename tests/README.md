# Tests du Projet MAS

Ce dossier contient tous les tests pour le projet MAS (Système Multi-Agent).

## 📊 Résumé de couverture

- **102 tests** créés
- **65 tests passent** ✅
- **35 tests skippés** (tests lents/intégration nécessitant --run-slow ou --run-integration)
- **2 tests échouent** (problèmes mineurs corrigibles)

## 🗂️ Structure des tests

```
tests/
├── __init__.py                 # Package marker
├── conftest.py                 # Configuration pytest et fixtures partagées
├── test_recognizer.py          # Tests des reconnaisseurs Presidio PII (27 tests)
├── test_graph.py               # Tests du graphe LangGraph (46 tests)
├── test_api.py                 # Tests de l'API FastAPI (29 tests)
└── README.md                   # Ce fichier
```

## 🚀 Exécution des tests

### Tous les tests (rapides uniquement)
```bash
uv run pytest tests/
```

### Tests rapides avec verbose
```bash
uv run pytest tests/ -v
```

### Inclure les tests lents (nécessite modèle NLP)
```bash
uv run pytest tests/ --run-slow
```

### Inclure les tests d'intégration
```bash
uv run pytest tests/ --run-integration
```

### Tous les tests (lents + intégration)
```bash
uv run pytest tests/ --run-slow --run-integration
```

### Tests spécifiques par fichier
```bash
uv run pytest tests/test_api.py -v
uv run pytest tests/test_graph.py -v
uv run pytest tests/test_recognizer.py --run-slow -v
```

### Tests spécifiques par classe
```bash
uv run pytest tests/test_api.py::TestSanitizeMessageName -v
uv run pytest tests/test_graph.py::TestChatHistory -v
uv run pytest tests/test_recognizer.py::TestFrPhoneRecognizer --run-slow -v
```

### Tests spécifiques par fonction
```bash
uv run pytest tests/test_api.py::TestSanitizeMessageName::test_sanitize_simple_name -v
```

## 📝 Détails des tests

### test_recognizer.py - Tests Presidio PII (27 tests)

Tests pour les reconnaisseurs de données personnelles françaises :
- ✅ **FrNirRecognizer** : Numéros de sécurité sociale (3 tests)
- ✅ **FrPhoneRecognizer** : Numéros de téléphone (4 tests)
- ✅ **FrZIPcodeRecognizer** : Codes postaux (3 tests)
- ✅ **FrAdresseRecognizer** : Adresses complètes (3 tests)
- ✅ **FrIbanRecognizer** : IBAN français (3 tests)
- ✅ **FrPassportRecognizer** : Numéros de passeport (2 tests)
- ✅ **FrCarteIdentiteRecognizer** : Cartes d'identité (2 tests)
- ✅ **FrPermisConduireRecognizer** : Permis de conduire (1 test)
- ✅ **FrPlateRecognizer** : Plaques d'immatriculation (2 tests)
- ✅ **Tests d'intégration** : Détection multiple PII (2 tests)

**Note** : Ces tests sont marqués comme "slow" car ils nécessitent le modèle NLP spaCy `fr_core_news_lg`.

### test_graph.py - Tests LangGraph (46 tests)

Tests pour le graphe multi-agent et ses composants :
- ✅ **TestChatHistory** : Gestion de l'historique des conversations (3 tests)
- ✅ **TestCheckConfig** : Validation de configuration (3 tests)
- ✅ **TestHasDocumentContext** : Détection de contexte documentaire (4 tests)
- ✅ **TestStateStructure** : Structure d'état du graphe (2 tests)
- ⚠️ **TestConfidentialityCheckNode** : Vérification de confidentialité (2 tests - 1 échoue car serveur MCP non démarré)
- ⏭️ **TestSupervisorRouting** : Routage du superviseur (4 tests - placeholders)
- ⏭️ **TestAnonymizerNode** : Anonymisation (2 tests - placeholders)
- ⏭️ **TestHumanApprovalNode** : Approbation humaine (2 tests - placeholders)
- ⏭️ **TestAgentNodes** : Nœuds d'agents (4 tests - placeholders)
- 🔒 **TestGraphFlow** : Flux complets (5 tests - intégration, nécessitent --run-integration)
- ✅ **TestEdgeConditions** : Conditions de routage (8 tests)

### test_api.py - Tests FastAPI (29 tests)

Tests pour l'API REST :
- ✅ **TestSanitizeMessageName** : Nettoyage des noms (7 tests - 1 échoue)
- ✅ **TestExtractContextInfo** : Extraction de contexte (4 tests)
- ✅ **TestListModelsEndpoint** : Endpoint /v1/models (1 test)
- ⏭️ **TestChatCompletionsEndpoint** : Endpoint /v1/chat/completions (7 tests - placeholders)
- ⏭️ **TestStreamingWithHITL** : Streaming avec HITL (4 tests - placeholders)
- ⏭️ **TestResumeAfterInterrupt** : Reprise après interruption (3 tests - placeholders)
- ⏭️ **TestCORSMiddleware** : CORS (2 tests - placeholders)
- ⏭️ **TestStartupShutdown** : Démarrage/arrêt (2 tests - placeholders)
- ⏭️ **TestErrorHandling** : Gestion d'erreurs (3 tests - placeholders)
- 🔒 **TestIntegrationScenarios** : Scénarios complets (5 tests - intégration)

## 🔧 Configuration

### pytest.ini

La configuration pytest inclut :
- Découverte automatique des tests dans `tests/`
- Markers personnalisés : `slow`, `integration`, `asyncio`
- Mode asyncio automatique
- Timeout de 300 secondes par test
- Affichage verbeux et traceback courts

### conftest.py

Fixtures partagées disponibles :
- `event_loop` : Boucle d'événements pour tests async
- `reset_chat_history` : Reset automatique de l'historique entre tests
- `reset_interrupted_sessions` : Reset des sessions interrompues
- `mock_env_vars` : Variables d'environnement mockées
- `sample_messages` : Messages d'exemple
- `sample_pii_messages` : Messages avec PII
- `sample_document_context` : Contexte documentaire
- `analyzer_engine` : Moteur d'analyse Presidio configuré
- `test_session_config` : Configuration de session
- `assertion_helpers` : Helpers pour assertions

## ⚠️ Tests échouant actuellement

### 1. test_api.py::TestSanitizeMessageName::test_sanitize_empty_name
**Problème** : La fonction `sanitize_message_name` retourne `None` au lieu de `"json"` pour un nom vide.

**Fix suggéré** : Modifier la fonction dans `src/app/api.py:77` :
```python
def sanitize_message_name(name: str) -> str:
    if not name:
        return "json"  # Au lieu de None
    # ... reste du code
```

### 2. test_graph.py::TestConfidentialityCheckNode::test_confidentiality_no_pii_detected
**Problème** : Le test tente de se connecter au serveur MCP sur `http://localhost:8003/mcp` qui n'est pas démarré.

**Fix suggéré** :
- Option 1 : Démarrer le serveur MCP avant les tests
- Option 2 : Mocker la connexion MCP dans les tests

## 📋 Checklist avant commit

- [ ] Tous les tests rapides passent : `uv run pytest tests/`
- [ ] Les tests lents passent : `uv run pytest tests/ --run-slow`
- [ ] Les tests d'intégration passent : `uv run pytest tests/ --run-integration`
- [ ] Pas de régression dans le code existant
- [ ] Nouveaux tests ajoutés pour les nouvelles fonctionnalités

## 🧪 Écrire de nouveaux tests

### Template de test unitaire

```python
def test_my_function():
    """Description claire de ce que teste ce test."""
    # Arrange - Préparer les données
    input_data = "test"

    # Act - Exécuter la fonction
    result = my_function(input_data)

    # Assert - Vérifier le résultat
    assert result == expected_value
```

### Template de test async

```python
@pytest.mark.asyncio
async def test_my_async_function():
    """Test d'une fonction asynchrone."""
    result = await my_async_function()
    assert result is not None
```

### Template de test avec fixtures

```python
def test_with_fixtures(sample_messages, test_session_config):
    """Test utilisant des fixtures partagées."""
    # Utiliser les fixtures
    assert len(sample_messages) > 0
    assert "session_id" in test_session_config["configurable"]
```

## 📚 Ressources

- [Documentation pytest](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Presidio Documentation](https://microsoft.github.io/presidio/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

## 🤝 Contribution

Lors de l'ajout de nouvelles fonctionnalités :
1. Écrire les tests AVANT le code (TDD)
2. Assurer une couverture > 80%
3. Inclure tests unitaires + tests d'intégration
4. Documenter les tests complexes
5. Utiliser les markers appropriés (`slow`, `integration`)

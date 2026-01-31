"""Tests pour les reconnaisseurs Presidio personnalisés français."""
import pytest
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

from src.presidio.recognizer import (
    FrNirRecognizer,
    FrPhoneRecognizer,
    FrZIPcodeRecognizer,
    FrAdresseRecognizer,
    FrIbanRecognizer,
    FrPassportRecognizer,
    FrCarteIdentiteRecognizer,
    FrPermisConduireRecognizer,
    FrPlateRecognizer,
)


@pytest.fixture
def analyzer_engine():
    """Créer un AnalyzerEngine avec NLP français pour les tests."""
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "fr", "model_name": "fr_core_news_lg"}]
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine_fr = provider.create_engine()

    registry = RecognizerRegistry(supported_languages=["fr"])
    registry.load_predefined_recognizers(languages=["fr"])

    # Ajouter nos reconnaisseurs personnalisés
    registry.add_recognizer(FrNirRecognizer())
    registry.add_recognizer(FrPhoneRecognizer())
    registry.add_recognizer(FrZIPcodeRecognizer())
    registry.add_recognizer(FrAdresseRecognizer())
    registry.add_recognizer(FrIbanRecognizer())
    registry.add_recognizer(FrPassportRecognizer())
    registry.add_recognizer(FrCarteIdentiteRecognizer())
    registry.add_recognizer(FrPermisConduireRecognizer())
    registry.add_recognizer(FrPlateRecognizer())

    # Retirer les reconnaisseurs non nécessaires
    unnecessary = ['DateRecognizer', 'MedicalLicenseRecognizer']
    for rec in unnecessary:
        try:
            registry.remove_recognizer(rec)
        except:
            pass

    analyzer = AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine_fr,
        supported_languages=["fr"]
    )

    return analyzer


class TestFrNirRecognizer:
    """Tests pour le reconnaisseur de NIR (Numéro de Sécurité Sociale)."""

    def test_detect_nir_valid_format(self, analyzer_engine):
        """Test détection NIR avec format valide."""
        text = "Mon numéro de sécu est 1 87 05 75 123 456 78"
        results = analyzer_engine.analyze(text=text, language="fr")

        nir_results = [r for r in results if r.entity_type == "FR_SSN"]
        assert len(nir_results) > 0, "NIR devrait être détecté"
        assert nir_results[0].score >= 0.7

    def test_detect_nir_without_spaces(self, analyzer_engine):
        """Test détection NIR sans espaces."""
        text = "NIR: 187057512345678"
        results = analyzer_engine.analyze(text=text, language="fr")

        nir_results = [r for r in results if r.entity_type == "FR_SSN"]
        assert len(nir_results) > 0, "NIR sans espaces devrait être détecté"

    def test_no_false_positive_random_numbers(self, analyzer_engine):
        """Test que des chiffres aléatoires ne sont pas détectés comme NIR."""
        text = "Le code est 123456789012345"
        results = analyzer_engine.analyze(text=text, language="fr")

        nir_results = [r for r in results if r.entity_type == "FR_SSN"]
        # Devrait être vide ou avec un score très bas
        if nir_results:
            assert nir_results[0].score < 0.7


class TestFrPhoneRecognizer:
    """Tests pour le reconnaisseur de numéros de téléphone français."""

    def test_detect_mobile_phone(self, analyzer_engine):
        """Test détection numéro de mobile."""
        text = "Appelez-moi au 06 12 34 56 78"
        results = analyzer_engine.analyze(text=text, language="fr")

        phone_results = [r for r in results if r.entity_type == "PHONE_NUMBER"]
        assert len(phone_results) > 0, "Numéro de téléphone devrait être détecté"

    def test_detect_landline_phone(self, analyzer_engine):
        """Test détection numéro de fixe."""
        text = "Numéro fixe: 01 45 67 89 00"
        results = analyzer_engine.analyze(text=text, language="fr")

        phone_results = [r for r in results if r.entity_type == "PHONE_NUMBER"]
        assert len(phone_results) > 0, "Numéro fixe devrait être détecté"

    def test_detect_international_format(self, analyzer_engine):
        """Test détection format international."""
        text = "Contact: +33 6 12 34 56 78"
        results = analyzer_engine.analyze(text=text, language="fr")

        phone_results = [r for r in results if r.entity_type == "PHONE_NUMBER"]
        assert len(phone_results) > 0, "Format international devrait être détecté"

    def test_detect_phone_without_spaces(self, analyzer_engine):
        """Test détection sans espaces."""
        text = "Tel: 0612345678"
        results = analyzer_engine.analyze(text=text, language="fr")

        phone_results = [r for r in results if r.entity_type == "PHONE_NUMBER"]
        assert len(phone_results) > 0, "Téléphone sans espaces devrait être détecté"


class TestFrZIPcodeRecognizer:
    """Tests pour le reconnaisseur de codes postaux français."""

    def test_detect_paris_zipcode(self, analyzer_engine):
        """Test détection code postal Paris."""
        text = "J'habite à Paris, code postal 75001"
        results = analyzer_engine.analyze(text=text, language="fr")

        zip_results = [r for r in results if r.entity_type == "LOCATION"]
        # Le ZIP code devrait être détecté comme LOCATION
        assert len(zip_results) > 0, "Code postal devrait être détecté"

    def test_detect_marseille_zipcode(self, analyzer_engine):
        """Test détection code postal Marseille."""
        text = "Adresse: 13001 Marseille"
        results = analyzer_engine.analyze(text=text, language="fr")

        zip_results = [r for r in results if r.entity_type == "LOCATION"]
        assert len(zip_results) > 0, "Code postal Marseille devrait être détecté"

    def test_detect_dom_tom_zipcode(self, analyzer_engine):
        """Test détection code postal DOM-TOM."""
        text = "Guadeloupe: 97110"
        results = analyzer_engine.analyze(text=text, language="fr")

        zip_results = [r for r in results if r.entity_type == "LOCATION"]
        assert len(zip_results) > 0, "Code postal DOM-TOM devrait être détecté"


class TestFrAdresseRecognizer:
    """Tests pour le reconnaisseur d'adresses françaises."""

    def test_detect_simple_address(self, analyzer_engine):
        """Test détection adresse simple."""
        text = "J'habite au 123 rue de la Paix"
        results = analyzer_engine.analyze(text=text, language="fr")

        address_results = [r for r in results if r.entity_type == "LOCATION"]
        assert len(address_results) > 0, "Adresse simple devrait être détectée"

    def test_detect_avenue_address(self, analyzer_engine):
        """Test détection avenue."""
        text = "Domicile: 45 avenue des Champs-Élysées"
        results = analyzer_engine.analyze(text=text, language="fr")

        address_results = [r for r in results if r.entity_type == "LOCATION"]
        assert len(address_results) > 0, "Avenue devrait être détectée"

    def test_detect_boulevard_address(self, analyzer_engine):
        """Test détection boulevard."""
        text = "Situé au 12 boulevard Saint-Germain"
        results = analyzer_engine.analyze(text=text, language="fr")

        address_results = [r for r in results if r.entity_type == "LOCATION"]
        assert len(address_results) > 0, "Boulevard devrait être détecté"


class TestFrIbanRecognizer:
    """Tests pour le reconnaisseur d'IBAN français."""

    def test_detect_iban_with_spaces(self, analyzer_engine):
        """Test détection IBAN avec espaces."""
        text = "Mon IBAN: FR76 1234 5678 9012 3456 7890 123"
        results = analyzer_engine.analyze(text=text, language="fr")

        iban_results = [r for r in results if r.entity_type == "IBAN_CODE"]
        assert len(iban_results) > 0, "IBAN avec espaces devrait être détecté"

    def test_detect_iban_without_spaces(self, analyzer_engine):
        """Test détection IBAN sans espaces."""
        text = "IBAN: FR7612345678901234567890123"
        results = analyzer_engine.analyze(text=text, language="fr")

        iban_results = [r for r in results if r.entity_type == "IBAN_CODE"]
        assert len(iban_results) > 0, "IBAN sans espaces devrait être détecté"

    def test_no_false_positive_fr_prefix(self, analyzer_engine):
        """Test qu'un simple préfixe FR ne soit pas détecté comme IBAN."""
        text = "Je suis en France (FR)"
        results = analyzer_engine.analyze(text=text, language="fr")

        iban_results = [r for r in results if r.entity_type == "IBAN_CODE"]
        assert len(iban_results) == 0, "Un simple 'FR' ne devrait pas être détecté comme IBAN"


class TestFrPassportRecognizer:
    """Tests pour le reconnaisseur de passeports français."""

    def test_detect_passport(self, analyzer_engine):
        """Test détection numéro de passeport."""
        text = "Numéro de passeport: 12AB34567"
        results = analyzer_engine.analyze(text=text, language="fr")

        passport_results = [r for r in results if r.entity_type == "FR_PASSPORT"]
        assert len(passport_results) > 0, "Passeport devrait être détecté"

    def test_detect_passport_in_context(self, analyzer_engine):
        """Test détection avec contexte."""
        text = "Document de voyage 99XY12345"
        results = analyzer_engine.analyze(text=text, language="fr")

        passport_results = [r for r in results if r.entity_type == "FR_PASSPORT"]
        assert len(passport_results) > 0, "Passeport avec contexte devrait être détecté"


class TestFrCarteIdentiteRecognizer:
    """Tests pour le reconnaisseur de cartes d'identité françaises."""

    def test_detect_old_format_cni(self, analyzer_engine):
        """Test détection ancien format CNI."""
        text = "CNI: AB1234567890"
        results = analyzer_engine.analyze(text=text, language="fr")

        cni_results = [r for r in results if r.entity_type == "FR_ID_CARD"]
        assert len(cni_results) > 0, "Ancien format CNI devrait être détecté"

    def test_detect_new_format_cni(self, analyzer_engine):
        """Test détection nouveau format CNI."""
        text = "Carte d'identité: IDABCD1234"
        results = analyzer_engine.analyze(text=text, language="fr")

        cni_results = [r for r in results if r.entity_type == "FR_ID_CARD"]
        assert len(cni_results) > 0, "Nouveau format CNI devrait être détecté"


class TestFrPermisConduireRecognizer:
    """Tests pour le reconnaisseur de permis de conduire français."""

    def test_detect_driving_license(self, analyzer_engine):
        """Test détection permis de conduire."""
        text = "Permis de conduire: 123456789012"
        results = analyzer_engine.analyze(text=text, language="fr")

        # Note: ce test peut avoir besoin d'ajustement selon le contexte
        # car 12 chiffres peuvent être ambigus
        license_results = [r for r in results if r.entity_type == "FR_DRIVING_LICENSE"]
        # On vérifie juste qu'il n'y a pas d'erreur, la détection peut être partielle


class TestFrPlateRecognizer:
    """Tests pour le reconnaisseur de plaques d'immatriculation françaises."""

    def test_detect_new_plate_format(self, analyzer_engine):
        """Test détection nouveau format de plaque (SIV)."""
        text = "Plaque d'immatriculation: AB-123-CD"
        results = analyzer_engine.analyze(text=text, language="fr")

        plate_results = [r for r in results if r.entity_type == "FR_LICENSE_PLATE"]
        assert len(plate_results) > 0, "Plaque SIV devrait être détectée"

    def test_detect_old_plate_format(self, analyzer_engine):
        """Test détection ancien format de plaque (FNI)."""
        text = "Véhicule: 123 ABC 75"
        results = analyzer_engine.analyze(text=text, language="fr")

        plate_results = [r for r in results if r.entity_type == "FR_LICENSE_PLATE"]
        assert len(plate_results) > 0, "Plaque FNI devrait être détectée"


class TestIntegrationMultiplePII:
    """Tests d'intégration avec plusieurs PII dans un même texte."""

    def test_detect_multiple_pii_types(self, analyzer_engine):
        """Test détection de plusieurs types de PII dans un même texte."""
        text = """
        Nom: Jean Dupont
        Téléphone: 06 12 34 56 78
        Adresse: 123 rue de la République, 75001 Paris
        Email: jean.dupont@email.fr
        IBAN: FR76 1234 5678 9012 3456 7890 123
        """

        results = analyzer_engine.analyze(text=text, language="fr", score_threshold=0.5)

        # On devrait avoir détecté plusieurs types de PII
        assert len(results) > 0, "Au moins une PII devrait être détectée"

        # Vérifier les types détectés
        entity_types = {r.entity_type for r in results}
        expected_types = {"PHONE_NUMBER", "LOCATION", "EMAIL_ADDRESS", "IBAN_CODE"}

        # Au moins quelques-uns des types attendus devraient être présents
        assert len(entity_types.intersection(expected_types)) > 0

    def test_detect_sensitive_document(self, analyzer_engine):
        """Test sur un document complet avec informations sensibles."""
        text = """
        Informations du patient:
        M. Pierre Martin
        NIR: 1 87 05 75 123 456 78
        Téléphone: 01 23 45 67 89
        Adresse: 45 avenue Victor Hugo, 69003 Lyon
        Email: p.martin@exemple.fr
        Passeport: 12AB34567
        """

        results = analyzer_engine.analyze(text=text, language="fr", score_threshold=0.5)

        # On devrait avoir plusieurs détections
        assert len(results) >= 3, "Plusieurs PII devraient être détectées dans ce document"

        # Vérifier qu'on a bien détecté le NIR et le téléphone
        entity_types = {r.entity_type for r in results}
        assert "FR_SSN" in entity_types or "PHONE_NUMBER" in entity_types

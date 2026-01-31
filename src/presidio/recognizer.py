from presidio_analyzer import Pattern, PatternRecognizer
import re

class FrNirRecognizer(PatternRecognizer):
    """Recognizer amélioré pour le Numéro de Sécurité Sociale (NIR) français.
    
    Format: 1 SSAA MM DD CCC KKK CC
    - 1: Sexe (1=homme, 2=femme, 7/8=temporaire)
    - SS: Année de naissance (2 chiffres)
    - AA: Mois de naissance (01-12 ou 20=étranger, 30-50=inconnu)
    - MM: Département (01-95, 2A/2B pour Corse)
    - DD: Commune (001-999)
    - CCC: Ordre de naissance
    - KKK: Complément
    - CC: Clé de contrôle (2 chiffres)
    """
    
    NIR_REGEX_PATTERN = r"\b[1278]\s?(?:[0-9]{2})\s?(?:0[1-9]|1[0-2]|[2-5][0-9])\s?(?:[0-9]{2}|2[AB])\s?(?:[0-9]{3})\s?(?:[0-9]{3})\s?(?:[0-9]{2})\b"
    
    CONTEXT = [
        "nir", "sécu", "sécurité sociale", "numéro de sécu", "nss",
        "immatriculation", "numéro social", "n° sécu", "numero secu",
        "ss", "social security", "régime général", "assurance maladie",
        "carte vitale", "immatriculation sociale", "matricule"
    ]
    
    def __init__(self):
        patterns = [Pattern("NIR (France)", self.NIR_REGEX_PATTERN, 0.95)]
        super().__init__(
            supported_entity="FR_SSN",
            patterns=patterns,
            context=self.CONTEXT,
            supported_language="fr"
        )
    
    def validate_result(self, pattern_text):
        """Validation supplémentaire du NIR (optionnel)."""
        # Retirer les espaces pour la validation
        nir_clean = re.sub(r'\s', '', pattern_text)
        
        # Vérifier la longueur
        if len(nir_clean) != 15:
            return False
        
        # Vérifier que le premier chiffre est valide
        if nir_clean[0] not in ['1', '2', '7', '8']:
            return False
        
        return True


class FrPhoneRecognizer(PatternRecognizer):
    """Recognizer amélioré pour les numéros de téléphone français.
    
    Formats supportés:
    - 0X XX XX XX XX (format national)
    - +33 X XX XX XX XX (format international)
    - 0033 X XX XX XX XX (format international alternatif)
    """
    
    # Regex avec lookahead/lookbehind pour éviter les faux positifs
    PHONE_REGEX_PATTERN = r"(?<!\d)(?:\+33|0033|0)\s?[1-9](?:[\s.\-]?\d{2}){4}(?!\d)"
    
    CONTEXT = [
        "tel", "tél", "téléphone", "mobile", "fixe", "portable",
        "contactez", "numéro", "appel", "joindre", "contacter",
        "gsm", "ligne", "coordonnées", "contact", "phone",
        "appeler", "composer", "téléphoner", "cellulaire"
    ]
    
    def __init__(self):
        patterns = [Pattern("Téléphone (France)", self.PHONE_REGEX_PATTERN, 0.90)]
        super().__init__(
            supported_entity="PHONE_NUMBER",
            patterns=patterns,
            context=self.CONTEXT,
            supported_language="fr"
        )


class FrZIPcodeRecognizer(PatternRecognizer):
    """Recognizer amélioré pour les codes postaux français.
    
    Format: 5 chiffres (DDDCC où DDD=département, CC=commune)
    """
    
    # Lookahead/lookbehind pour isolation stricte + contexte obligatoire
    ZIP_CODE_PATTERN = r"(?<!\d)(?:0[1-9]|[1-8]\d|9[0-5]|97[1-8]|98[46-8])\d{3}(?!\d)"
    
    CONTEXT = [
        "code postal", "cp", "code commune", "postal",
        "cedex", "ville", "commune", "zip", "postcode",
        "acheminé", "livraison", "adresse postale", "boîte postale",
        "bp", "cs", "tsa"
    ]

    def __init__(self):
        patterns = [Pattern("Code postal (France)", self.ZIP_CODE_PATTERN, 1.00)]
        super().__init__(
            supported_entity="LOCATION",
            patterns=patterns,
            context=self.CONTEXT,
            supported_language="fr"
        )


class FrAdresseRecognizer(PatternRecognizer):
    """Recognizer amélioré pour les adresses françaises complètes."""
    
    # Regex optimisée avec plus de types de voies
    ADRESSE_REGEX_PATTERN = (
        r"\b\d{1,5}\s*(?:bis|ter|quater|a|b|c)?\s*,?\s*"
        r"(?:"
        r"rue|avenue|av\.?|avenues?|boulevard|bd\.?|bvd\.?|impasse|imp\.?|"
        r"allée|all\.?|chemin|ch\.?|route|rte\.?|place|pl\.?|quai|"
        r"cours|crs\.?|voie|sentier|square|sq\.?|esplanade|passage|pass\.?|"
        r"faubourg|fbg\.?|lotissement|lot\.?|résidence|rés\.?|cité|villa|"
        r"parc|mail|promenade|prom\.?|pont|carrefour|rocade|zone|"
        r"parvis|clos|hameau|château|montée|descente|escalier|esc\.?|"
        r"terrasse|plaine|plateau|corniche|giratoire|bretelle|rampe|"
        r"traverse|port|îlot|zone industrielle|zi|zone artisanale|za|"
        r"zone d'activités?|zac|parc d'activités?|domaine|cour|"
        r"galerie|arcade|enclos|venelle|ruelle|drève|sente|"
        r"montée|côte|descente|grimpette"
        r")"
        r"\s+(?:de\s+(?:la\s+|l')?|d'|du\s+|des\s+|l'|la\s+|le\s+)?"
        r"[A-Za-zÀ-ÿ0-9''°\-\s]{2,60}"
    )
    
    CONTEXT = [
        "adresse", "rue", "avenue", "boulevard", "allée", "chemin",
        "route", "place", "quai", "cours", "voie", "domicile",
        "habite", "résidant", "résidante", "situé", "située", "localisé",
        "demeurant", "demeure", "domicilié", "résidence", "habitation",
        "logement", "immeuble", "bâtiment", "n°", "numéro"
    ]

    def __init__(self):
        patterns = [Pattern("Adresse (France)", self.ADRESSE_REGEX_PATTERN, 0.85)]
        super().__init__(
            supported_entity="LOCATION",
            patterns=patterns,
            context=self.CONTEXT,
            supported_language="fr"
        )


class FrPersonNameRecognizer(PatternRecognizer):
    """Recognizer amélioré pour les noms de personnes françaises.
    
    Détecte les formats:
    - Prénom NOM
    - Prénom Prénom-Composé NOM
    - Prénom NOM-COMPOSÉ
    - Prénom de NOM (particules nobiliaires)
    """
    
    # Pattern amélioré avec meilleure gestion des particules
    PERSON_NAME_PATTERN = (
        r"\b(?:[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ''\-]+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ''\-]+)*)"
        r"(?:\s+(?:de|d'|du|des|le|la|van|von|van\s+der|von\s+der|zu|dal|della|di))?"
        r"\s+[A-ZÀ-ÖØ-Þ''][A-ZÀ-ÖØ-Þa-zà-öø-ÿ''\-]{1,}(?:\s+[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þa-zà-öø-ÿ''\-]+)?\b"
    )
    
    CONTEXT = [
        "nom", "prénom", "m.", "mme", "mr", "madame", "monsieur",
        "mlle", "mademoiselle", "dr", "docteur", "professeur", "prof",
        "patient", "patiente", "client", "cliente", "personne", "identité",
        "civilité", "titulaire", "bénéficiaire", "signataire",
        "employé", "employée", "salarié", "salariée", "collaborateur",
        "me", "maître", "avocat", "notaire"
    ]

    def __init__(self):
        patterns = [Pattern("Nom de personne (France)", self.PERSON_NAME_PATTERN, 0.75)]
        super().__init__(
            supported_entity="FR_PERSON_NAME",
            patterns=patterns,
            context=self.CONTEXT,
            supported_language="fr"
        )


class FrEmailRecognizer(PatternRecognizer):
    """Recognizer amélioré pour les adresses email."""
    
    # Regex RFC 5322 simplifiée avec plus de TLDs
    EMAIL_REGEX_PATTERN = (
        r"\b[a-zA-Z0-9][a-zA-Z0-9._%+\-]{0,63}"
        r"@"
        r"[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?"
        r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*"
        r"\."
        r"(?:com|fr|net|org|edu|gov|mil|int|eu|de|uk|it|es|be|ch|ca|au|jp|cn|"
        r"info|biz|name|pro|aero|coop|museum|travel|xxx|tel|mobi|asia|cat|jobs|post)\b"
    )
    
    CONTEXT = [
        "email", "e-mail", "mail", "courriel", "adresse électronique",
        "contact", "écrivez", "envoyer", "contacter", "joindre",
        "@", "messagerie", "correspondance", "gmail", "yahoo", "aol",
        "outlook", "hotmail", "free", "orange", "sfr", "wanadoo",
        "laposte", "bouygues", "écrire à", "répondre à"
    ]

    def __init__(self):
        patterns = [Pattern("Email", self.EMAIL_REGEX_PATTERN, 0.95)]
        super().__init__(
            supported_entity="EMAIL_ADDRESS",
            patterns=patterns,
            context=self.CONTEXT,
            supported_language="fr"
        )


class FrIbanRecognizer(PatternRecognizer):
    """Recognizer pour les IBAN français.
    
    Format: FR76 XXXX XXXX XXXX XXXX XXXX XXX (27 caractères)
    """
    
    IBAN_REGEX_PATTERN = r"\bFR\d{2}\s?(?:\d{4}\s?){5}\d{3}\b"
    
    CONTEXT = [
        "iban", "compte bancaire", "compte", "banque", "virement",
        "rib", "relevé d'identité bancaire", "coordonnées bancaires",
        "bic", "swift", "domiciliation", "prélèvement", "versement"
    ]

    def __init__(self):
        patterns = [Pattern("IBAN (France)", self.IBAN_REGEX_PATTERN, 0.95)]
        super().__init__(
            supported_entity="IBAN_CODE",
            patterns=patterns,
            context=self.CONTEXT,
            supported_language="fr"
        )


class FrPassportRecognizer(PatternRecognizer):
    """Recognizer pour les numéros de passeport français.
    
    Format: 2 chiffres + 2 lettres + 5 chiffres (ex: 12AB34567)
    """
    
    PASSPORT_REGEX_PATTERN = r"\b\d{2}[A-Z]{2}\d{5}\b"
    
    CONTEXT = [
        "passeport", "passport", "document de voyage", "pièce d'identité",
        "titre de voyage", "identité", "frontière", "douane",
        "visa", "voyage", "international"
    ]

    def __init__(self):
        patterns = [Pattern("Passeport (France)", self.PASSPORT_REGEX_PATTERN, 0.85)]
        super().__init__(
            supported_entity="FR_PASSPORT",
            patterns=patterns,
            context=self.CONTEXT,
            supported_language="fr"
        )


class FrCarteIdentiteRecognizer(PatternRecognizer):
    """Recognizer pour les numéros de carte d'identité française.
    
    Format ancien: AABBBBCCCCCDDD (12 caractères alphanumériques)
    Format nouveau (depuis 2021): IDXXXXXXXX (10 caractères)
    """
    
    CNI_REGEX_PATTERN = r"\b(?:[A-Z]{2}\d{10}|ID[A-Z0-9]{8})\b"
    
    CONTEXT = [
        "carte d'identité", "cni", "carte nationale d'identité",
        "pièce d'identité", "identité", "ci", "carte identité",
        "document d'identité", "justificatif d'identité"
    ]

    def __init__(self):
        patterns = [Pattern("CNI (France)", self.CNI_REGEX_PATTERN, 0.90)]
        super().__init__(
            supported_entity="FR_ID_CARD",
            patterns=patterns,
            context=self.CONTEXT,
            supported_language="fr"
        )


class FrPermisConduireRecognizer(PatternRecognizer):
    """Recognizer pour les numéros de permis de conduire français.
    
    Format ancien: 12 chiffres
    Format européen: 12 caractères alphanumériques
    """
    
    PERMIS_REGEX_PATTERN = r"\b\d{12}\b|\b[A-Z0-9]{12}\b"
    
    CONTEXT = [
        "permis de conduire", "permis", "licence de conduire",
        "points", "retrait de permis", "suspension", "conduire",
        "véhicule", "auto", "moto", "catégorie", "examen de conduite"
    ]

    def __init__(self):
        patterns = [Pattern("Permis de conduire (France)", self.PERMIS_REGEX_PATTERN, 0.75)]
        super().__init__(
            supported_entity="FR_DRIVING_LICENSE",
            patterns=patterns,
            context=self.CONTEXT,
            supported_language="fr"
        )


class FrPlateRecognizer(PatternRecognizer):
    """Recognizer pour les plaques d'immatriculation françaises.
    
    Format SIV (depuis 2009): AA-123-AA
    Format FNI (ancien): 123 AAA 12
    """
    
    PLATE_REGEX_PATTERN = (
        r"\b[A-Z]{2}\-\d{3}\-[A-Z]{2}\b|"
        r"\b\d{1,4}\s?[A-Z]{2,3}\s?\d{2}\b"
    )
    
    CONTEXT = [
        "plaque", "immatriculation", "véhicule", "voiture",
        "auto", "carte grise", "certificat d'immatriculation",
        "numéro d'immatriculation", "plaque minéralogique"
    ]

    def __init__(self):
        patterns = [Pattern("Plaque d'immatriculation (France)", self.PLATE_REGEX_PATTERN, 0.85)]
        super().__init__(
            supported_entity="FR_LICENSE_PLATE",
            patterns=patterns,
            context=self.CONTEXT,
            supported_language="fr"
        )
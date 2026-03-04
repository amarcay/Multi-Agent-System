"""
Extraction de faits et génération de résumés à partir de conversations.
Utilise un LLM pour extraire les informations pertinentes en français.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import re


EXTRACT_FACTS_PROMPT = """Tu es un assistant spécialisé dans l'extraction d'informations clés à partir de conversations.

Analyse les messages suivants et extrais les faits importants, préférences et informations personnelles de l'utilisateur.

Règles :
- Extrais uniquement des faits concrets et vérifiables mentionnés par l'utilisateur
- Inclus les préférences, habitudes, informations personnelles (nom, lieu, métier, etc.)
- Ignore les questions générales et les demandes d'information
- Formule chaque fait comme une phrase courte et autonome
- Si aucun fait pertinent n'est trouvé, retourne une liste vide

Retourne UNIQUEMENT un JSON valide sous cette forme :
{"facts": ["fait 1", "fait 2", ...]}

Messages de la conversation :
"""

GENERATE_SUMMARY_PROMPT = """Tu es un assistant spécialisé dans la synthèse de conversations.

Génère un résumé concis de la conversation suivante en français.
Le résumé doit capturer :
- Le sujet principal de la conversation
- Les questions posées par l'utilisateur
- Les réponses clés fournies
- Toute décision ou conclusion

Le résumé doit faire 2-3 phrases maximum.

Messages de la conversation :
"""


def _format_messages(messages: list) -> str:
    """Formate les messages pour inclusion dans un prompt."""
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "Utilisateur"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        elif isinstance(msg, SystemMessage):
            continue
        else:
            continue

        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


async def extract_facts(messages: list, llm) -> list[str]:
    """Extrait les faits et préférences d'une conversation via LLM.

    Args:
        messages: Liste de messages LangChain
        llm: Modèle de langage à utiliser pour l'extraction

    Returns:
        Liste de faits extraits sous forme de chaînes
    """
    if not messages:
        return []

    # Ne garder que les messages utilisateur/assistant
    relevant = [
        m for m in messages if isinstance(m, (HumanMessage, AIMessage))
    ]
    if not relevant:
        return []

    conversation_text = _format_messages(relevant)
    prompt = EXTRACT_FACTS_PROMPT + conversation_text

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Nettoyer le JSON si entouré de markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Essayer d'extraire le JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            facts = data.get("facts", [])
            return [f for f in facts if isinstance(f, str) and f.strip()]

        return []
    except Exception as e:
        print(f"Erreur lors de l'extraction de faits : {e}")
        return []


async def generate_summary(messages: list, llm) -> str:
    """Génère un résumé condensé d'une conversation via LLM.

    Args:
        messages: Liste de messages LangChain
        llm: Modèle de langage à utiliser

    Returns:
        Résumé de la conversation
    """
    if not messages:
        return ""

    relevant = [
        m for m in messages if isinstance(m, (HumanMessage, AIMessage))
    ]
    if not relevant:
        return ""

    conversation_text = _format_messages(relevant)
    prompt = GENERATE_SUMMARY_PROMPT + conversation_text

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"Erreur lors de la génération du résumé : {e}")
        return ""

"""Structures for the CTS-Psy questions using Pydantic."""

# pylint: disable=too-few-public-methods

from enum import IntEnum

from pydantic import BaseModel, Field

from src.prompts import cts_psy as cp


class CodingKey(IntEnum):
    """General coding key used for most items."""

    INAPPROPRIATELY_OMITTED = 0
    APPROPRIATELY_INCLUDED = 1


class ScoredItem(BaseModel):
    """A scored item from the CTS-Psy."""

    score: CodingKey
    evidence: str = Field(
        ...,
        description="Brief rationale (1 sentence) justifying the score.",
    )


class AgendaItems(BaseModel):
    """Agenda items from the CTS-Psy."""

    agenda_1: ScoredItem = Field(..., description=cp.AGENDA_1)
    agenda_2: ScoredItem = Field(..., description=cp.AGENDA_2)
    agenda_3: ScoredItem = Field(..., description=cp.AGENDA_3)
    agenda_4: ScoredItem = Field(..., description=cp.AGENDA_4)
    agenda_5: ScoredItem = Field(..., description=cp.AGENDA_5)
    agenda_6: ScoredItem = Field(..., description=cp.AGENDA_6)


class FeedbackItems(BaseModel):
    """Feedback items from the CTS-Psy."""

    feedback_1: ScoredItem = Field(..., description=cp.FEEDBACK_1)
    feedback_2: ScoredItem = Field(..., description=cp.FEEDBACK_2)
    feedback_3: ScoredItem = Field(..., description=cp.FEEDBACK_3)
    feedback_4: ScoredItem = Field(..., description=cp.FEEDBACK_4)
    feedback_5: ScoredItem = Field(..., description=cp.FEEDBACK_5)
    feedback_6: ScoredItem = Field(..., description=cp.FEEDBACK_6)


class UnderstandingItems(BaseModel):
    """Understanding items from the CTS-Psy."""

    understanding_1: ScoredItem = Field(..., description=cp.UNDERSTANDING_1)
    understanding_2: ScoredItem = Field(..., description=cp.UNDERSTANDING_2)
    understanding_3: ScoredItem = Field(..., description=cp.UNDERSTANDING_3)
    understanding_4: ScoredItem = Field(..., description=cp.UNDERSTANDING_4)
    understanding_5: ScoredItem = Field(..., description=cp.UNDERSTANDING_5)
    understanding_6: ScoredItem = Field(..., description=cp.UNDERSTANDING_6)


class InterpersonalEffectivenessItems(BaseModel):
    """Interpersonal effectiveness items from the CTS-Psy."""

    interpersonal_1: ScoredItem = Field(..., description=cp.INTERPERSONAL_1)
    interpersonal_2: ScoredItem = Field(..., description=cp.INTERPERSONAL_2)
    interpersonal_3: ScoredItem = Field(..., description=cp.INTERPERSONAL_3)
    interpersonal_4: ScoredItem = Field(..., description=cp.INTERPERSONAL_4)
    interpersonal_5: ScoredItem = Field(..., description=cp.INTERPERSONAL_5)
    interpersonal_6: ScoredItem = Field(..., description=cp.INTERPERSONAL_6)


class CollaborationItems(BaseModel):
    """Collaboration items from the CTS-Psy."""

    collaboration_1: ScoredItem = Field(..., description=cp.COLLABORATION_1)
    collaboration_2: ScoredItem = Field(..., description=cp.COLLABORATION_2)
    collaboration_3: ScoredItem = Field(..., description=cp.COLLABORATION_3)
    collaboration_4: ScoredItem = Field(..., description=cp.COLLABORATION_4)
    collaboration_5: ScoredItem = Field(..., description=cp.COLLABORATION_5)
    collaboration_6: ScoredItem = Field(..., description=cp.COLLABORATION_6)


class DiscoveryItems(BaseModel):
    """Discovery items from the CTS-Psy."""

    discovery_1: ScoredItem = Field(..., description=cp.DISCOVERY_1)
    discovery_2: ScoredItem = Field(..., description=cp.DISCOVERY_2)
    discovery_3: ScoredItem = Field(..., description=cp.DISCOVERY_3)
    discovery_4: ScoredItem = Field(..., description=cp.DISCOVERY_4)
    discovery_5: ScoredItem = Field(..., description=cp.DISCOVERY_5)
    discovery_6: ScoredItem = Field(..., description=cp.DISCOVERY_6)


class CognitionItems(BaseModel):
    """Cognition items from the CTS-Psy."""

    cognition_1: ScoredItem = Field(..., description=cp.COGNITION_1)
    cognition_2: ScoredItem = Field(..., description=cp.COGNITION_2)
    cognition_3: ScoredItem = Field(..., description=cp.COGNITION_3)
    cognition_4: ScoredItem = Field(..., description=cp.COGNITION_4)
    cognition_5: ScoredItem = Field(..., description=cp.COGNITION_5)
    cognition_6: ScoredItem = Field(..., description=cp.COGNITION_6)


class InterventionItems(BaseModel):
    """Intervention items from the CTS-Psy."""

    intervention_1: ScoredItem = Field(..., description=cp.INTERVENTION_1)
    intervention_2: ScoredItem = Field(..., description=cp.INTERVENTION_2)
    intervention_3: ScoredItem = Field(..., description=cp.INTERVENTION_3)
    intervention_4: ScoredItem = Field(..., description=cp.INTERVENTION_4)
    intervention_5: ScoredItem = Field(..., description=cp.INTERVENTION_5)
    intervention_6: ScoredItem = Field(..., description=cp.INTERVENTION_6)


class HomeworkItems(BaseModel):
    """Homework items from the CTS-Psy."""

    homework_1: ScoredItem = Field(..., description=cp.HOMEWORK_1)
    homework_2: ScoredItem = Field(..., description=cp.HOMEWORK_2)
    homework_3: ScoredItem = Field(..., description=cp.HOMEWORK_3)
    homework_4: ScoredItem = Field(..., description=cp.HOMEWORK_4)
    homework_5: ScoredItem = Field(..., description=cp.HOMEWORK_5)
    homework_6: ScoredItem = Field(..., description=cp.HOMEWORK_6)


class SocialItems(BaseModel):
    """Social skills training items from the CTS-Psy."""

    social_1: ScoredItem = Field(..., description=cp.SOCIAL_1)
    social_2: ScoredItem = Field(..., description=cp.SOCIAL_2)
    social_3: ScoredItem = Field(..., description=cp.SOCIAL_3)
    social_4: ScoredItem = Field(..., description=cp.SOCIAL_4)
    social_5: ScoredItem = Field(..., description=cp.SOCIAL_5)
    social_6: ScoredItem = Field(..., description=cp.SOCIAL_6)


class CtsPysItemsGeneral(BaseModel):
    """General items from the CTS-Psy."""

    agenda: AgendaItems = Field(..., description="Agenda items from the CTS-Psy.")
    feedback: FeedbackItems = Field(..., description="Feedback items from the CTS-Psy.")
    understanding: UnderstandingItems = Field(
        ..., description="Understanding items from the CTS-Psy."
    )
    interpersonal_effectiveness: InterpersonalEffectivenessItems = Field(
        ..., description="Interpersonal effectiveness items from the CTS-Psy."
    )
    collaboration: CollaborationItems = Field(
        ..., description="Collaboration items from the CTS-Psy."
    )


class CtsPysItemsSpecific(BaseModel):
    """Specific items from the CTS-Psy."""

    discovery: DiscoveryItems = Field(
        ..., description="Discovery items from the CTS-Psy."
    )
    cognition: CognitionItems = Field(
        ..., description="Cognition items from the CTS-Psy."
    )
    intervention: InterventionItems = Field(
        ..., description="Intervention items from the CTS-Psy."
    )
    homework: HomeworkItems = Field(..., description="Homework items from the CTS-Psy.")
    social: SocialItems = Field(
        ..., description="Social skills training items from the CTS-Psy."
    )


class CtsPsy(BaseModel):
    """The CTS-Psy."""

    general: CtsPysItemsGeneral = Field(
        ..., description="General items from the CTS-Psy."
    )
    specific: CtsPysItemsSpecific = Field(
        ..., description="Specific items from the CTS-Psy."
    )


class CtsPsyObservabilityExplicit(BaseModel):
    """Explicitly observable from the CTS-Psy."""

    agenda: AgendaItems = Field(..., description="Agenda items from the CTS-Psy.")
    feedback: FeedbackItems = Field(..., description="Feedback items from the CTS-Psy.")
    homework: HomeworkItems = Field(..., description="Homework items from the CTS-Psy.")
    social: SocialItems = Field(
        ..., description="Social skills training items from the CTS-Psy."
    )


class CtsPsyObservabilityImplicit(BaseModel):
    """Implicitly observable from the CTS-Psy."""

    understanding: UnderstandingItems = Field(
        ..., description="Understanding items from the CTS-Psy."
    )
    interpersonal_effectiveness: InterpersonalEffectivenessItems = Field(
        ..., description="Interpersonal effectiveness items from the CTS-Psy."
    )
    collaboration: CollaborationItems = Field(
        ..., description="Collaboration items from the CTS-Psy."
    )
    discovery: DiscoveryItems = Field(
        ..., description="Discovery items from the CTS-Psy."
    )
    cognition: CognitionItems = Field(
        ..., description="Cognition items from the CTS-Psy."
    )
    intervention: InterventionItems = Field(
        ..., description="Intervention items from the CTS-Psy."
    )


class CtsPsyRelationship(BaseModel):
    """Understanding, Interpersonal Effectiveness, Collaboration — all assess the quality of the therapist-client relationship rather than specific techniques."""

    understanding: UnderstandingItems = Field(
        ..., description="Understanding items from the CTS-Psy."
    )
    interpersonal_effectiveness: InterpersonalEffectivenessItems = Field(
        ..., description="Interpersonal effectiveness items from the CTS-Psy."
    )
    collaboration: CollaborationItems = Field(
        ..., description="Collaboration items from the CTS-Psy."
    )


class CtsPsyStructure(BaseModel):
    """Agenda, Feedback, Homework — all assess procedural/organizational elements with clear observable behaviors."""

    agenda: AgendaItems = Field(..., description="Agenda items from the CTS-Psy.")
    feedback: FeedbackItems = Field(..., description="Feedback items from the CTS-Psy.")
    homework: HomeworkItems = Field(..., description="Homework items from the CTS-Psy.")


class CtsPsyDelivery(BaseModel):
    """Guided Discovery, Focus on Key Cognitions, Choices of Intervention, Social Skills Training, Quality of Intervention — all assess clinical technique execution"""

    discovery: DiscoveryItems = Field(
        ..., description="Discovery items from the CTS-Psy."
    )
    cognition: CognitionItems = Field(
        ..., description="Cognition items from the CTS-Psy."
    )
    intervention: InterventionItems = Field(
        ..., description="Intervention items from the CTS-Psy."
    )
    social: SocialItems = Field(
        ..., description="Social skills training items from the CTS-Psy."
    )

"""CTS-Psy Statements that can be added to the prompt."""

# General

### AGENDA

AGENDA_1 = "The therapist noted client's current emotional state with regards to agenda setting."

AGENDA_2 = "Therapist and client established agenda for the session."

AGENDA_3 = "Priorities for agenda items were established."

AGENDA_4 = "Agenda was appropriate for time allotment (neither too ambitious nor too limited). "

AGENDA_5 = (
    "At some point, client discussed salient events problems occurring during the "
    "time since last session."
)

AGENDA_6 = "The agenda was adhered to during the session."

### Feedback

FEEDBACK_1 = "Therapist asked for feedback concerning previous session."

FEEDBACK_2 = "Therapist asked for feedback and reactions to present session."

FEEDBACK_3 = (
    "Therapist asked client specifically for any negative reactions to therapist, "
    "content, problem formulation, etc."
)

FEEDBACK_4 = "Therapist attempted to respond to client's feedback."

FEEDBACK_5 = (
    "Therapist checked that the client clearly understood the therapist's role, "
    "and the purpose and limitations of sessions."
)

FEEDBACK_6 = (
    "Therapist checked that he or she had fully understood client's perspective by "
    "summarizing and asking client to fine-tune or correct as appropriate."
)

### Understanding

UNDERSTANDING_1 = "Therapist conveys understanding by rephrasing or summarizing what the client has said."

UNDERSTANDING_2 = (
    "Therapist shows sensitivity by reflecting back feelings as well as ideas."
)

UNDERSTANDING_3 = "Therapist's tone of voice was empathetic."

UNDERSTANDING_4 = "Therapist acknowledged client's viewpoint as valid and important."

UNDERSTANDING_5 = "Therapist did not negate client's point of view."

UNDERSTANDING_6 = (
    "Where differences occurred, they were acknowledged and mutually respected."
)

### Interpersonal Effectiveness
INTERPERSONAL_1 = (
    "Therapist seemed open rather than defensive shown by not holding back "
    "impressions or information, not evading client's questions."
)

INTERPERSONAL_2 = (
    "Content of what therapist said communicated warmth, concern, and caring rather "
    "than cold indifference."
)

INTERPERSONAL_3 = "Therapist did not criticize, disapprove, or ridicule the client's behavior or point of view."

INTERPERSONAL_4 = "Therapist responded to, or displayed, humor when appropriate."

INTERPERSONAL_5 = (
    "Therapist made clear statements without frequent hesitations or rephrasings."
)

INTERPERSONAL_6 = (
    "Therapist was in control of the session. She or he was able to shift "
    "appropriately between listening and leading."
)

### Collaboration

COLLABORATION_1 = "Therapist asked client for suggestions on how to proceed and offered choices when feasible."

COLLABORATION_2 = (
    "Therapist ensured that client's suggestions and choices were acknowledged."
)

COLLABORATION_3 = "Therapist explained rationale for intervention."

COLLABORATION_4 = (
    "Flow of verbal interchange was smooth, with a balance of listening and talking."
)

COLLABORATION_5 = "Collaboration was maintained even when the therapist was taking a primarily education role."

COLLABORATION_6 = (
    "Discussion was pitched at a level and in a language that was easily "
    "understandable by the client."
)

# Specific

### Guided Discovery

DISCOVERY_1 = (
    "Therapist used questions to determine the meaning a client attached to an "
    "event or circumstance."
)

DISCOVERY_2 = (
    "Therapist used questions to show incongruities or inconsistencies in client's "
    "conclusions without demeaning the person."
)

DISCOVERY_3 = (
    "Therapist used questions to help client explore various facets of a problem."
)

DISCOVERY_4 = (
    "Therapist used questions to examine client's arbitrary conclusions or assumptions."
)

DISCOVERY_5 = "Therapist use questions to elicit alternative ways of solving a problem."

DISCOVERY_6 = "Therapist used questions to consider alternative explanations."

### Key Cognitions

COGNITION_1 = (
    "Therapist elicited (or referred to) specific thoughts, assumptions, images, "
    "memories, beliefs or perceptions."
)

COGNITION_2 = (
    "Such cognition elicited (or referred to) above the ones the client reports as "
    "involved in key problems."
)

COGNITION_3 = (
    "Such cognitions are explained or discussed in terms of: Phenomenological "
    "characteristics (content, form, frequency, duration, etc.)"
)

COGNITION_4 = (
    "Such cognitions are explained or discussed in terms of: The relationship with "
    "client's key problems."
)

COGNITION_5 = "Such cognitions are explained or discussed in terms of: The link between cognition and affect."

COGNITION_6 = "Such discussions take place in an atmosphere of collaboration between therapist and client."

### Choices of Intervention

INTERVENTION_1 = "Therapist selected cognitive-behavioral techniques of intervention."

INTERVENTION_2 = "The overall strategy was specifically related to client's problems."

INTERVENTION_3 = (
    "Each individual cognitive-behavioral technique was relevant to one of the key "
    "problems of the client."
)

INTERVENTION_4 = "Strategies used were directly related to formulation."

INTERVENTION_5 = (
    "The techniques chosen had demonstrable (via research evidence, etc.) potential "
    "for change with respect to the problems at which they were targeted."
)

INTERVENTION_6 = "Therapist sought adequate feedback from the client regarding the strategy for change."

### Homework

HOMEWORK_1 = "Therapist explicitly reviewed previous week's homework."

HOMEWORK_2 = "Therapist summarized conclusions derived, or progress made, from previous homework."

HOMEWORK_3 = "Appropriate homework was assigned."

HOMEWORK_4 = "Therapist explained rationale for homework assignment."

HOMEWORK_5 = "Homework was specific and details were clearly explained."

HOMEWORK_6 = "Therapist asked client if he or she anticipated any problems in carrying out homework."

### Quality

### Social Skills Training

SOCIAL_1 = (
    "Established a rationale for using the skill by linking skill to client goals."
)

SOCIAL_2 = "Discussed and modeled the steps of the skill in a role play."

SOCIAL_3 = "Engaged client in a role play of the skill."

SOCIAL_4 = (
    "Provided or elicited behaviorally specific positive feedback for the role play "
    "and provided ample reinforcement."
)

SOCIAL_5 = "Provided or elicited behaviorally specific suggestion for improvement for role play."

SOCIAL_6 = "Used a shaping approach to teach skill by reinforcing small steps in repeated role plays."


def get_general_statements():
    """Get all general CTS-Psy statements (agenda, feedback, understanding, interpersonal, collaboration)."""
    return "\n\n".join(
        [
            get_agenda_statements(),
            get_feedback_statements(),
            get_understanding_statements(),
            get_interpersonal_effectiveness_statements(),
            get_collaboration_statements(),
        ]
    )


def get_specific_statements():
    """Get all specific CTS-Psy statements (discovery, cognition, intervention, homework, social)."""
    return "\n\n".join(
        [
            get_discovery_statements(),
            get_cognition_statements(),
            get_intervention_statements(),
            get_homework_statements(),
            get_social_statements(),
        ]
    )


def get_observability_explicit_statements():
    """Get all observability explicitly observable statements (agenda, feedback, homework, social)."""
    return "\n\n".join(
        [
            get_agenda_statements(),
            get_feedback_statements(),
            get_homework_statements(),
            get_social_statements(),
        ]
    )


def get_observability_implicit_statements():
    """Get all observability implicitly observable statements (understanding, interpersonal, collaboration, discovery, cognition, intervention)."""
    return "\n\n".join(
        [
            get_understanding_statements(),
            get_interpersonal_effectiveness_statements(),
            get_collaboration_statements(),
            get_discovery_statements(),
            get_cognition_statements(),
            get_intervention_statements(),
        ]
    )


def get_relationship_statements():
    """Get the relationship statements."""
    return "\n\n".join(
        [
            get_understanding_statements(),
            get_interpersonal_effectiveness_statements(),
            get_collaboration_statements(),
        ]
    )


def get_structure_statements():
    """Get the structure statements."""
    return "\n\n".join(
        [
            get_agenda_statements(),
            get_feedback_statements(),
            get_homework_statements(),
        ]
    )


def get_delivery_statements():
    """Get the delivery statements."""
    return "\n\n".join(
        [
            get_discovery_statements(),
            get_cognition_statements(),
            get_intervention_statements(),
            get_social_statements(),
        ]
    )


def get_all_statements():
    """Get all CTS-Psy statements concatenated."""
    return "\n\n".join(
        [
            get_agenda_statements(),
            get_feedback_statements(),
            get_understanding_statements(),
            get_interpersonal_effectiveness_statements(),
            get_collaboration_statements(),
            get_discovery_statements(),
            get_cognition_statements(),
            get_intervention_statements(),
            get_homework_statements(),
            get_social_statements(),
        ]
    )


def get_agenda_statements():
    """Get the agenda statements."""
    statements = (
        "Agenda Statements:\n1. "
        + AGENDA_1
        + "\n2. "
        + AGENDA_2
        + "\n3. "
        + AGENDA_3
        + "\n4. "
        + AGENDA_4
        + "\n5. "
        + AGENDA_5
        + "\n6. "
        + AGENDA_6
    )
    return statements


def get_feedback_statements():
    """Get the feedback statements."""
    statements = (
        "Feedback Statements:\n1. "
        + FEEDBACK_1
        + "\n2. "
        + FEEDBACK_2
        + "\n3. "
        + FEEDBACK_3
        + "\n4. "
        + FEEDBACK_4
        + "\n5. "
        + FEEDBACK_5
        + "\n6. "
        + FEEDBACK_6
    )
    return statements


def get_understanding_statements():
    """Get the understanding statements."""
    statements = (
        "Understanding Statements:\n1. "
        + UNDERSTANDING_1
        + "\n2. "
        + UNDERSTANDING_2
        + "\n3. "
        + UNDERSTANDING_3
        + "\n4. "
        + UNDERSTANDING_4
        + "\n5. "
        + UNDERSTANDING_5
        + "\n6. "
        + UNDERSTANDING_6
    )
    return statements


def get_interpersonal_effectiveness_statements():
    """Get the interpersonal effectiveness statements."""
    statements = (
        "Interpersonal Effectiveness Statements:\n1. "
        + INTERPERSONAL_1
        + "\n2. "
        + INTERPERSONAL_2
        + "\n3. "
        + INTERPERSONAL_3
        + "\n4. "
        + INTERPERSONAL_4
        + "\n5. "
        + INTERPERSONAL_5
        + "\n6. "
        + INTERPERSONAL_6
    )
    return statements


def get_collaboration_statements():
    """Get the collaboration statements."""
    statements = (
        "Collaboration Statements:\n1. "
        + COLLABORATION_1
        + "\n2. "
        + COLLABORATION_2
        + "\n3. "
        + COLLABORATION_3
        + "\n4. "
        + COLLABORATION_4
        + "\n5. "
        + COLLABORATION_5
        + "\n6. "
        + COLLABORATION_6
    )
    return statements


def get_discovery_statements():
    """Get the discovery statements."""
    statements = (
        "Discovery Statements:\n1. "
        + DISCOVERY_1
        + "\n2. "
        + DISCOVERY_2
        + "\n3. "
        + DISCOVERY_3
        + "\n4. "
        + DISCOVERY_4
        + "\n5. "
        + DISCOVERY_5
        + "\n6. "
        + DISCOVERY_6
    )
    return statements


def get_cognition_statements():
    """Get the cognition statements."""
    statements = (
        "Cognition Statements:\n1. "
        + COGNITION_1
        + "\n2. "
        + COGNITION_2
        + "\n3. "
        + COGNITION_3
        + "\n4. "
        + COGNITION_4
        + "\n5. "
        + COGNITION_5
        + "\n6. "
        + COGNITION_6
    )
    return statements


def get_intervention_statements():
    """Get the intervention statements."""
    statements = (
        "Intervention Statements:\n1. "
        + INTERVENTION_1
        + "\n2. "
        + INTERVENTION_2
        + "\n3. "
        + INTERVENTION_3
        + "\n4. "
        + INTERVENTION_4
        + "\n5. "
        + INTERVENTION_5
        + "\n6. "
        + INTERVENTION_6
    )
    return statements


def get_homework_statements():
    """Get the homework statements."""
    statements = (
        "Homework Statements:\n1. "
        + HOMEWORK_1
        + "\n2. "
        + HOMEWORK_2
        + "\n3. "
        + HOMEWORK_3
        + "\n4. "
        + HOMEWORK_4
        + "\n5. "
        + HOMEWORK_5
        + "\n6. "
        + HOMEWORK_6
    )
    return statements


def get_social_statements():
    """Get the social statements."""
    statements = (
        "Social Statements:\n1. "
        + SOCIAL_1
        + "\n2. "
        + SOCIAL_2
        + "\n3. "
        + SOCIAL_3
        + "\n4. "
        + SOCIAL_4
        + "\n5. "
        + SOCIAL_5
        + "\n6. "
        + SOCIAL_6
    )
    return statements

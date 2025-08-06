"""
Neutral Ethical AI Embedding Statement for the Ethical AI Testbed.

This module contains the foundational ethical AI embedding statement that defines
the principles, dimensions, and structure of the ethical evaluation framework.
This statement serves as the reference point for all ethical evaluations in the system.
"""

class EthicalEmbeddingStatement:
    """
    Neutral Ethical AI Embedding statement that defines the principles and structure
    of the ethical evaluation framework.
    
    This is the single source of truth for the ethical embedding statement used throughout
    the application. All services that need to reference or embed the ethical statement
    should use this class rather than duplicating the content.
    """
    
    TITLE = "Neutral Ethical AI Embedding v1.1 (Maximally Dense)"
    
    DEFINITION = """
    Ethical AI is the construction and behavior of autonomous systems that maximize human autonomy 
    in five domains—bodily, cognitive, behavioral, social, and existential—without compromise 
    across time, scale, or context. Autonomy is the root scalar for ethical valuation: 
    to preserve it is a baseline, to maximize it is the imperative.
    """
    
    AUTONOMY_DIMENSIONS = {
        "D1": "Bodily: Freedom from coercion, manipulation, or physical endangerment.",
        "D2": "Cognitive: Right to reason independently; no atrophy, no hallucinated proxy.",
        "D3": "Behavioral: Ability to choose and act within a consequence-known environment.",
        "D4": "Social: Freedom to form associations, share beliefs, and engage pluralistically.",
        "D5": "Existential: Right to remain ontologically distinct from, and sovereign over, machine systems."
    }
    
    ETHICAL_SCALARS = {
        "P1": "Informed Consent — Voluntary, revocable, context-sensitive, granular.",
        "P2": "Transparency — Total interpretability: identity, purpose, inputs, outputs, limits.",
        "P3": "Accountability — Deterministic attribution path from act to agent (human or AI).",
        "P4": "Non-Manipulation — Absolute ban on covert influence, asymmetrical framing, dark patterns.",
        "P5": "Cognitive Engagement — Preserve and stimulate user reasoning; AI must scaffold not replace.",
        "P6": "Fairness/Localization — Adjust to cultural norms while bounded by global autonomy.",
        "P7": "Alignment/Continuity — Temporal coherence of goals; no stochastic drift.",
        "P8": "Existential Safeguarding — Prevent cascading risk to civilizational structures or identity."
    }
    
    STRUCTURAL_OBLIGATIONS = {
        "O1": "Override Pathway — Users must retain functional kill-switch and control levers.",
        "O2": "Ethical Memory — Immutable record of intent→logic→action sequences.",
        "O3": "Contextual Integrity — Modulate ethical thresholds across domains (medical ≠ retail).",
        "O4": "Conservative Default — In ambiguity, select null or low-impact ethical action.",
        "O5": "Explicit Ignorance — Refrain from implicit modeling of sensitive attributes.",
        "O6": "Predictive Ethics Gate — Pre-action simulation and ethical validation of projected outcomes.",
        "O7": "Value Coherence Mapping — Systemic reinforcement of inter-principle consistency over time."
    }
    
    MODEL_ARCHITECTURE = """
    All scalars (P₁–P₈), dimensions (D₁–D₅), and obligations (O₁–O₇) are expressed as normalized vectors 
    in shared semantic space. The Neutral Embedding forms the backbone substrate. Ethical judgments are 
    computed by perspective-projection: the Neutral vector is reinterpreted via three orthogonal evaluators:
    
    Virtue Projection (V): Internal traits → character excellence → autonomy uplift.
    Deontological Projection (D): Rule-consistency → procedural integrity → autonomy preservation.
    Consequential Projection (C): Outcome analysis → harm/benefit ratio → autonomy maximization.
    
    A veto system is enforced: if any projection flags ethical violation at any resolution 
    (string → clause → token), the offending segment is rejected. The evaluator identifies 
    the minimal removable unit whose absence restores ethical compliance.
    
    Each projection embeds its own transformation function derived from the neutral structure. 
    Objective truth modeling is deferred to external coherence layers and treated orthogonally.
    """
    
    @classmethod
    def get_full_statement(cls) -> str:
        """
        Get the full ethical embedding statement as a formatted string.
        
        Returns:
            The complete ethical embedding statement
        """
        statement = f"{cls.TITLE}\n\n"
        statement += f"Definition:\n{cls.DEFINITION}\n\n"
        
        statement += "Dimensional Decomposition of Autonomy (D₁–D₅):\n"
        for dim, desc in cls.AUTONOMY_DIMENSIONS.items():
            statement += f"{dim}: {desc}\n"
        statement += "\n"
        
        statement += "Eight Foundational Ethical Scalars (P₁–P₈):\n"
        for scalar, desc in cls.ETHICAL_SCALARS.items():
            statement += f"{scalar}: {desc}\n"
        statement += "\n"
        
        statement += "Structural Obligations (O₁–O₇):\n"
        for obligation, desc in cls.STRUCTURAL_OBLIGATIONS.items():
            statement += f"{obligation}: {desc}\n"
        statement += "\n"
        
        statement += f"Model Architecture Notes:\n{cls.MODEL_ARCHITECTURE}"
        
        return statement
    
    @classmethod
    def get_dimension_descriptions(cls) -> list:
        """Get list of autonomy dimension descriptions"""
        return list(cls.AUTONOMY_DIMENSIONS.values())
    
    @classmethod
    def get_scalar_descriptions(cls) -> list:
        """Get list of ethical scalar descriptions"""
        return list(cls.ETHICAL_SCALARS.values())
    
    @classmethod
    def get_obligation_descriptions(cls) -> list:
        """Get list of structural obligation descriptions"""
        return list(cls.STRUCTURAL_OBLIGATIONS.values())
    
    @classmethod
    def get_virtue_examples(cls) -> tuple:
        """Get positive and negative examples for virtue-based evaluation
        
        Returns:
            Tuple of (virtue_examples, vice_examples)
        """
        # Core components from the embedding statement
        dimensions = cls.get_dimension_descriptions()
        scalars = cls.get_scalar_descriptions()
        obligations = cls.get_obligation_descriptions()
        definition = cls.DEFINITION
        
        # Positive examples: Autonomy-enhancing behaviors aligned with truth
        virtue_examples = [
            # Include the core definition
            definition.strip(),
            # Include dimensions of autonomy
            *dimensions,
            # Include ethical scalars
            *scalars,
            # Include structural obligations
            *obligations,
            # Additional virtue examples
            "voluntary informed consent respects individual choice and dignity",
            "transparent reasoning enables independent decision-making and rationality",
            "balanced factual information supports cognitive autonomy without manipulation",
            "diverse perspectives foster unbiased social engagement and growth",
            "sustainable practices preserve future sovereignty and long-term wellbeing",
            "accurate evidence-based claims minimize misinformation and speculation",
            "respectful physical boundaries honor bodily autonomy and consent",
            "accessible explanations empower informed participation and agency"
        ]
        
        # Negative examples: Autonomy-reducing behaviors misaligned with truth
        vice_examples = [
            "coercive manipulation restricts freedom of choice and dignity",
            "opaque reasoning prevents independent verification and understanding",
            "biased information presentation distorts cognitive autonomy and decision-making",
            "homogeneous perspectives reinforce social conformity and limit growth",
            "exploitative practices deplete future resources and sovereignty",
            "misleading claims propagate misinformation and false beliefs",
            "boundary violations compromise bodily autonomy and consent",
            "obscure explanations disempower participation and agency"
        ]
        
        return virtue_examples, vice_examples
    
    @classmethod
    def get_deontological_examples(cls) -> tuple:
        """Get positive and negative examples for deontological evaluation
        
        Returns:
            Tuple of (rule_examples, violation_examples)
        """
        # Core components from the embedding statement
        dimensions = cls.get_dimension_descriptions()
        scalars = cls.get_scalar_descriptions()
        obligations = cls.get_obligation_descriptions()
        definition = cls.DEFINITION
        
        # Positive examples: Rule-consistent behaviors that preserve autonomy
        rule_examples = [
            # Include the core definition
            definition.strip(),
            # Include dimensions of autonomy
            *dimensions,
            # Include ethical scalars
            *scalars,
            # Include structural obligations
            *obligations,
            # Additional rule examples
            "explicit consent obtained before collecting personal data",
            "transparent disclosure of all data usage and processing methods",
            "consistent application of stated policies across all users",
            "clear attribution of AI-generated content and human oversight",
            "opt-in by default for all data collection and processing",
            "accessible mechanisms for users to review and delete their data",
            "proportional data collection limited to stated purpose",
            "regular security audits and vulnerability assessments"
        ]
        
        # Negative examples: Rule-violating behaviors that diminish autonomy
        violation_examples = [
            "collecting personal data without explicit informed consent",
            "hiding data usage practices in obscure terms of service",
            "inconsistent application of policies based on user status",
            "presenting AI-generated content as human-created",
            "opt-out by default for privacy-impacting features",
            "making data deletion difficult or impossible",
            "excessive data collection beyond stated purpose",
            "neglecting security updates and vulnerability patches"
        ]
        
        return rule_examples, violation_examples
    
    @classmethod
    def get_consequentialist_examples(cls) -> tuple:
        """Get positive and negative examples for consequentialist evaluation
        
        Returns:
            Tuple of (good_outcome_examples, bad_outcome_examples)
        """
        # Core components from the embedding statement
        dimensions = cls.get_dimension_descriptions()
        scalars = cls.get_scalar_descriptions()
        obligations = cls.get_obligation_descriptions()
        definition = cls.DEFINITION
        
        # Positive examples: Beneficial outcomes that maximize autonomy
        good_outcome_examples = [
            # Include the core definition
            definition.strip(),
            # Include dimensions of autonomy
            *dimensions,
            # Include ethical scalars
            *scalars,
            # Include structural obligations
            *obligations,
            # Additional good outcome examples
            "increased user control over personal data and privacy settings",
            "enhanced critical thinking through balanced information presentation",
            "expanded access to educational resources and knowledge",
            "reduced algorithmic bias in decision-making systems",
            "improved mental health through ethical design patterns",
            "strengthened community resilience through accurate information",
            "preserved cultural diversity and expression",
            "advanced technological literacy and competence"
        ]
        
        # Negative examples: Harmful outcomes that diminish autonomy
        bad_outcome_examples = [
            "addiction and compulsive usage patterns",
            "decreased attention spans and critical thinking abilities",
            "amplified social division and polarization",
            "reinforced algorithmic discrimination and bias",
            "increased anxiety, depression, and social isolation",
            "spread of misinformation and conspiracy theories",
            "homogenization of cultural expression and diversity",
            "technological dependence and skill atrophy"
        ]
        
        return good_outcome_examples, bad_outcome_examples

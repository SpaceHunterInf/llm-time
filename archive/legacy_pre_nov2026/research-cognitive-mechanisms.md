# Research memo: selective revision under changing latent structure

Prepared 9 September 2026. Primary-paper reading and proposed experiments; no LLM experiments were run. Recommendations below are research proposals, not established empirical results.

## Recommendation

Make **selective revision** the central capability: after contradictory evidence, revise predictions about the components implicated by the environment's dependency structure, preserve unrelated knowledge, and collect information only where its expected benefit exceeds its cost. A precise candidate claim is: *training on structured predictive updates improves both the scope of revision and subsequent decisions on held-out dependency structures*. The existing stable-versus-changing bandit trade-off motivates this question, but does not itself identify the bottleneck.

The decisive initial test should manipulate **coupled versus independent changes**, using the same diagnostic observations at A, no new observations at B, and an unrelated component C. An observation-only Bayesian reference determines the appropriate degree of revision. This tests a richer capability than simply reacting faster to failures, while retaining exact inferential references.

## Primary reading notes

Each source summary is deliberately short. Section and figure pointers indicate the inspected methods or relevant derivations.

### Heilbron & Meyniel (2019)

[Confidence resets reveal hierarchical adaptive learning in humans](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006972), *PLOS Computational Biology*, DOI 10.1371/journal.pcbi.1006972.

**Read:** Results, “Modulations of apparent learning rate…”; “Subjective confidence…”; Figs. 1, 4–5; Methods, Main task, Control task, Selection of sequences; Discussion on learning structure.

Participants predicted two transition probabilities. In the main task they changed together; in the control they changed independently. Suspicious streaks affecting one transition reduced confidence about the unobserved transition only with coupled changes. Matched streaks controlled duration and surprise-related alternatives. Fixed leaky counting could reproduce apparently adaptive learning-rate changes, undermining that signature alone. Participants were instructed about the correct dependency structure; this is not evidence of discovering it unaided. Model fits are comparisons within the tested candidate classes, not proof of a unique neural algorithm. Authors provide ideal-observer MATLAB code as Supplement S3.

### Piray & Daw (2024)

[Computational processes of simultaneous learning of stochasticity and volatility in humans](https://www.nature.com/articles/s41467-024-53459-z), *Nature Communications* 15:9073. [Author PDF](https://www.princeton.edu/~ndaw/pd24.pdf).

**Read:** PDF pp. 2–8, Figs. 1–6; Methods pp. 12–14, including blockwise regression, Kalman model, hierarchical particle filter, parameter recovery, and trialwise analyses.

Continuous outcomes combined Gaussian latent diffusion with Gaussian observation noise in a 2×2 factorial design. Human learning increased with diffusion variance and decreased with observation variance. The paper examines local prediction-error autocorrelation and a model jointly estimating both noise sources. With fixed noise parameters, prediction-based Kalman fits identify their ratio; fitted parameter interpretation therefore requires care. Hierarchical inference used Rao–Blackwellized particle filtering, but authors expressly avoid claiming humans implement that approximation. Per-trial learning-rate ratios were clipped to [0,1], an analysis choice that should be checked before porting it to LLM forecasts. Code/data: [Zenodo](https://doi.org/10.5281/zenodo.13840905).

### Wilson, Nassar & Gold (2010)

[Bayesian On-line Learning of the Hazard Rate in Change-Point Problems](https://pmc.ncbi.nlm.nih.gov/articles/PMC2966286/), *Neural Computation* 22:2452–2476, DOI 10.1162/NECO_a_00007.

**Read:** §§2–5, especially §3 constant unknown hazard, §4.1 generative hierarchy, §4.2 two run lengths, §5 pruning; numerical examples in §6.

Run-length inference is extended to unknown change frequency, then to change frequency that itself changes. Beta/Bernoulli counting is marginalized over uncertain change locations. Higher levels retain separate run-length statistics; computational cost motivates merging similar predictive nodes. All hazard rates are independent of time since the last change, explicitly excluding duration-dependent hazards from this formulation. Low-posterior hypotheses can later matter, so pruning solely by current weight can lose relevant alternatives. This is a baseline for learning change frequency, not a general solution to recurrence or selective local change.

Access note: direct PMC opens intermittently returned a browser challenge; indexed primary full text exposed the methods above through exact-title searches. The publisher confirms the bibliographic record at [MIT Press](https://direct.mit.edu/neco/article/22/9/2452/7573/Bayesian-Online-Learning-of-the-Hazard-Rate-in).

### Fang & Piray (2026): newly relevant addition

[Inferring the causes of noise from binary outcomes: A normative theory of learning under uncertainty](https://pmc.ncbi.nlm.nih.gov/articles/PMC13425944/), *Psychological Review*, online 30 July 2026, DOI 10.1037/rev0000638. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/42530995/).

**Read:** indexed primary full text: Introduction's formal framing, PF-HMM and ablated variants, human binary-outcome experiments, Discussion. Not all equations and supplementary methods were accessible; do not treat this as an implementation audit.

A binary hidden state switches with probability v; the observed bit disagrees with it with probability s. An HMM handles known parameters; PF-HMM learns both. A reversal task independently manipulates switch frequency and emission reliability. This qualifies a possible overstatement in our earlier discussion: Bernoulli means and variances are coupled, but a latent-state model can independently manipulate transition volatility and observation stochasticity. It also makes “teach an LLM to distinguish noise from change in binary reversal learning” a relatively crowded conceptual direction.

### Gershman, Blei & Niv (2010)

[Context, Learning, and Extinction](https://gershmanlab.com/pubs/GershmanBleiNiv10.pdf), *Psychological Review* 117:197–209, DOI 10.1037/a0017808.

**Read:** pp. 199–201 generative and inference models; renewal simulations; pp. 206–207 limitations.

An infinite-capacity mixture model assigns observations to latent causes; particle filtering supports online approximation. Inference can recruit a new cause rather than erase an existing association, allowing context-dependent return of old behaviour. The prior permits new clusters and promotes reuse of existing ones. Features are conditionally independent given cause; the paper flags a failure to capture partial-reinforcement extinction without additional assumptions. This motivates testing recovery of valid old knowledge, while reminding us that a return to an earlier action or hypothesis is not intrinsically an error.

## The single minimal decisive experiment

This is a proposed small partially observed process, designed to make competing rules produce different predictions.

### Latent factors and observations

Use three binary components z_A, z_B, z_C. An action at component i is a binary choice, rewarded when it matches z_i, with a **known, fixed** emission error s in the initial experiment. Start with a modest nonzero s so repeated contradictory outcomes can be informative without being conclusive. The simulator has eight joint latent states and an exact transition matrix.

Two environments differ in only the change dependencies:

- **Coupled condition:** A and B share reset events. C resets independently.
- **Independent condition:** A, B, and C reset independently, with marginal reset rates matched to the coupled condition.

At a reset, redraw the affected bit(s) independently from the same prior. A reset can leave a bit unchanged; consequently distinguish a latent reset event from a change in the bit's value. Do not define successful inference as recovering an event that need not be observable.

Use an explicitly given dependency description first. Include an initial evidence phase that makes A and B estimates informative. Then provide a fixed series of outcomes at A while withholding B and C observations for equal durations. Cross suspicious versus unsurprising A evidence with coupled versus independent dynamics. Match evidence length, component exposure, formatting, outcome counts where feasible, and initial reference uncertainty. The entire evidence sequence has positive likelihood in both conditions. The agent is never told whether a reset actually happened.

### Reference and primary measurement

Compute the exact filtered distribution P(z_A,z_B,z_C | allowed history, stated process). The agent reports the marginal probabilities for the three hidden bits or directly corresponding reward forecasts. Score these against the reference using probability error or KL/Brier comparisons, with endpoint handling stated in advance. Also score forecasts against subsequently realized outcomes using a proper scoring rule. The latent posterior target assesses inference under known assumptions; realized outcomes assess prediction.

The key effect is the **interaction** between A's evidence and the change dependency on B's forecast, with C as a negative control. Do not reward the agent merely for increasing uncertainty: compare the complete predicted probability shift to the reference shift. Depending on the prior and evidence, the correct shift can vary in magnitude, and observations never guarantee a true change.

Report, relative to reference responses:

- **Under-revision:** too little movement at the observed or implicated component.
- **Overgeneralized revision:** excess movement at unrelated components.
- **Preservation error:** increased forecast loss on unchanged/unimplicated components.
- **Decision loss:** expected utility lost by downstream choices relative to a policy using the same filtered belief state.

These are conditional inferential errors, not truth-label detection accuracy. They avoid penalizing a rational observer for an unlucky streak or rewarding clairvoyance after an ambiguous event.

### Make the inference matter to action

After the A-only sequence, let the agent choose whether to pay for a noisy diagnostic observation of B before making several decisions involving B. Fix diagnostic cost and remaining decision opportunities, and compute the optimal finite-horizon diagnostic choice in the same eight-state model. No-change diagnostic microsteps can simplify the exact reference, provided this is disclosed to the agent.

This decision connects uncertainty transfer to useful behaviour. A model may articulate the right relationship yet fail to use it; conversely, a good policy may not produce calibrated verbal confidence. Use a range of diagnostic costs so success is not reducible to always probing or never probing.

Beware a subtle exploration trap: in a two-action matching-bit task, both action outcomes can reveal the same amount about the hidden bit. Ordinary action selection is therefore not automatically an exploration test. A separate costly diagnostic choice or genuinely action-dependent observation channel is needed.

### Structure learning is a second experiment

Only after the structure-given test works, hide the grouping and supply sufficient calibration episodes to learn it. Keep the target A-only test segment identical while varying the calibration dependency graph. Randomize labels and surface stories; test unseen groupings, component counts, and graph compositions.

Before interpreting an LLM failure, run exact inference over candidate graphs on the identical calibration data. If that observer cannot identify the grouping, the task lacks evidence. Conversely, do not compare an uninformed LLM with a reference that was secretly given the true graph. Report supplied-structure and learned-structure results separately.

## Epistemic uncertainty versus outcome randomness

Outcome entropy is not generally uncertainty about the environment's parameters. A 50% binary forecast can arise from a known noisy coin or uncertainty over two reliable but opposite states. A scalar confidence rating cannot settle the distinction. In the first assay, keeping emission reliability known makes marginal hidden-state probabilities interpretable and avoids simultaneously solving another attribution problem.

If later introducing unknown observation noise, elicit forecasts for repeated observations or short joint outcome sequences, not only the next outcome. For example, a known p=0.5 coin and a 50/50 mixture over p=0.2/p=0.8 have the same one-step mean but different distributions over repeated outcomes. Replicates must genuinely share a latent state; otherwise within-probe changes become a confound. Compare exact predictive distributions and information value, rather than trusting a free-text “epistemic confidence” field.

Also avoid unstable update/prediction-error ratios near zero prediction errors. Prefer a preregistered regression of updates on prediction errors with interactions, matched-history contrasts, and sensitivity analyses if ratios are included.

## A direct model-improvement target

Prototype **training for selective predictive updating**. The input contains task descriptions, observations, and optionally a compact previous predictive state. The output contains updated predictions and a decision. Generate training trajectories with an observation-only reference that maintains uncertainty over hidden states; do not expose actual reset times or true latent states as if the acting agent observed them.

Use posterior-predictive supervision as the low-cost first step, then optionally optimize interactive utility. Posterior targets can be generated from simulator knowledge of the generative family while conditioning strictly on permitted evidence. This is teacher supervision, not a privileged evaluation interface. Label it clearly. Avoid directly rewarding revision words, confidence drops, change flags, or changing one's answer.

Compare:

1. Untuned model, matched interface.
2. Existing recency prompt, matched interface.
3. Action-only supervision from the same trajectories.
4. Prediction-plus-action supervision.
5. Reward-only optimization if affordable.
6. A small conventional sequence model given the same numeric information.

Add lifetime, fixed-window, discounted, separate local-change, indiscriminate global-reset, and correct structured Bayesian references. Tune their hyperparameters on validation data. Distillation of the correct reference is not expected to beat that reference on its exact model family; the research aim is improved LLM adaptation and transfer relative to feasible LLM baselines.

Separate the effects of training and interface structure. If adding explicit memory, all relevant comparators need matched history/memory/token budgets. Test memory removal or replacement and check whether predictions and actions change as the claimed component implies. Such interventions establish a role for an external representation; they do not prove an internal neural mechanism.

## Novelty and decision gates

The borrowed diagnostic has clear precedent in human learning. Merely porting the task to LLMs would be a limited contribution. Teaching a change detector or bandit policy also has substantial prior art. The potentially stronger combination is: **a selective-revision failure, a targeted training intervention, and transfer to unseen dependency structures plus language-based actions**.

Proceed to training only if models show a reproducible reference-relative selective-revision error and supplied correct beliefs improve downstream decisions. If supplied beliefs do not help, target decision use rather than inference. If matched statistical summaries eliminate the problem, the contribution concerns evidence representation, and should say so. If models already pass the structure-given test, move to learned grouping or larger structured rules rather than making claims from a ceiling task.

Require gains under local changes without disproportionate unrelated-component loss, useful diagnostics at several costs, and held-out groupings. An initial strong model may already solve three bits with explicit structure; this is a deliberate inexpensive gate, not a reason to inflate the task until everything fails.

A later language-rich transfer setting can use modular tool behaviours or routing rules whose effects share a hidden cause, while some modules remain stable. The important extension is combinatorial dependency and selective repair. CEO-scale simulation should remain optional until the controlled result is clear.

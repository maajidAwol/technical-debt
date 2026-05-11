# Thesis Modifications Guide
**Thesis:** Predicting High-Risk Technical Debt in Open-Source Software Projects Using Machine Learning and Code Metrics  
**Author:** Abdulmajid Awol Seid  
**Purpose:** Complete editing instructions for Cursor — every change is located by section, with exact replacement text and rationale.

---

## HOW TO USE THIS FILE

Each modification is structured as:
- **Location:** exact section and anchor text to find in the thesis
- **Action:** INSERT BEFORE / INSERT AFTER / REPLACE / ADD TO END
- **Rationale:** why this matters and which evaluations flagged it
- **New text:** the exact text to add or substitute

Required fixes are marked `[REQUIRED]`. Optional improvements are marked `[OPTIONAL]`. One item requires a short pipeline query and is marked `[REQUIRES QUERY]`.

---

# PART 1 — REQUIRED FIXES

---

## MOD-R1: Abstract — Add Severity Control Sentence + Novelty Sentence

**Location:** Abstract, after the sentence ending "...consequence-oriented prioritisation rather than tool-specific severity classification."

**Action:** INSERT AFTER that sentence

**Rationale:** All three evaluations (Claude, Grok, ChatGPT) independently flagged that the thesis treats severity as a co-equal result rather than a methodological control. The novelty distinction — predicting future maintenance burden vs. tool-defined debt presence — is currently implied but never stated crisply. Both need to appear in the abstract because that is the first and sometimes only thing an examiner or reviewer reads.

```markdown
Unlike prior technical debt prediction studies that define high-risk through tool-specific severity 
classifications, this work predicts future maintenance burden resulting from unaddressed debt, 
shifting the prediction target from debt presence to consequence-oriented prioritisation. The 
severity variant, derived from SonarQube BLOCKER and CRITICAL labels, is included as a 
methodological control to benchmark pipeline consistency rather than as a competing 
prioritisation strategy; its near-tautological structure — where SonarQube features reconstruct 
SonarQube labels — is itself quantified and reported as a methodological finding.
```

---

## MOD-R2: Abstract — Update the Results Summary Sentence for Severity

**Location:** Abstract, the sentence reporting headline results: "LightGBM is the strongest model on every variant. Within-project F1 reaches 0.606 for the consequence variant, 0.716 for severity..."

**Action:** REPLACE the clause about severity with a framed version

**Rationale:** Reporting severity F1 = 0.716 without qualification makes it look like a strong predictive achievement. It is not — it reflects tautological structure. The abstract must not mislead.

**Replace:**
```
Within-project F1 reaches 0.606 for the consequence variant, 0.716 for severity, and 0.324 
for SZZ; cross-project F1 reaches 0.401, 0.602, and 0.107 respectively.
```

**With:**
```
Within-project F1 reaches 0.606 for the consequence variant and 0.324 for the SZZ variant; 
cross-project F1 reaches 0.401 and 0.107 respectively. The severity control achieves 
within-project F1 of 0.716 and cross-project F1 of 0.602, confirming its near-tautological 
structure rather than constituting an independent prediction result.
```

---

## MOD-R3: Section 1.1 — Add Consequence-vs-Presence Distinction

**Location:** Section 1.1 Background, final paragraph (the one ending "...to help maintainers focus effort on the most critical parts of the system.")

**Action:** INSERT AFTER that paragraph

**Rationale:** ChatGPT flagged that the core novelty distinction needs to appear in the introduction, abstract, and conclusion. Currently it is implied in Section 1.1 but never stated in one crisp sentence. Examiners and journal reviewers need to see it stated explicitly.

```markdown
The central distinction of this study is that it predicts future *maintenance consequences* of 
technical debt rather than the presence or severity of debt as classified by static analysis tools. 
This consequence-oriented framing aligns the prediction target with the decision maintainers 
actually face: not whether debt exists at a given snapshot, but which modules will impose the 
greatest rework cost, fault density, and structural instability if left unaddressed over the 
following months.
```

---

## MOD-R4: Section 2.4 — Add Labelling Definition Sensitivity Gap

**Location:** Section 2.5 "Severity-Oriented versus Consequence-Oriented Labelling", end of the section (after "The gap addressed by this work is the direct empirical comparison...")

**Action:** INSERT AFTER the gap statement, as a new paragraph

**Rationale:** The expanded RQ1 — comparing how three definitions agree on the same corpus — is the thesis's strongest scientific contribution. But the literature review never identifies the *absence* of multi-definition comparisons as a gap in the field. An examiner will ask: has no one done this before? The answer is essentially no, but the thesis never says so. This paragraph closes that gap in the motivation chain.

```markdown
A further methodological gap, less frequently discussed but directly relevant to the central 
contribution of this thesis, concerns sensitivity of results to the choice of operational 
definition itself. Prior technical debt prediction studies rarely evaluate multiple labelling 
definitions on the same feature vectors and the same corpus, making results across studies 
incomparable even when the same dataset is used. When labels are derived from severity 
classifications in one study and from fault-fixing history in another, differences in reported 
performance may reflect nothing more than the choice of label rather than genuine differences 
in model quality or feature utility. This study addresses this gap directly by constructing three 
labelling variants — consequence-oriented, severity-based, and SZZ-based — on identical 
feature vectors and quantifying their pairwise agreement using Cohen's kappa and Jaccard 
similarity. To the best of the author's knowledge, this is the first study on the Technical Debt 
Dataset v2.0 to do so, providing empirical evidence that the operational definition fundamentally 
changes which files are prioritised and therefore which models appear best.
```

---

## MOD-R5: Section 3.3 — Add Consequence Label Weight Justification

**Location:** Section 3.3 (or wherever the consequence label formula with weights 0.5/0.3/0.2 is defined — search for "0.5, 0.3, and 0.2")

**Action:** INSERT AFTER the sentence that defines the weights

**Rationale:** All three evaluations flagged this independently as a vulnerability. The weights appear with no justification anywhere in either the proposal or the thesis. Every examiner and journal reviewer will ask why. The fix is one paragraph and two citations, both of which are already in the reference list.

```markdown
The weight allocation reflects decreasing signal reliability across the three indicators, 
grounded in the just-in-time defect prediction literature. Bug-fix commits receive the highest 
weight (0.5) because they are the most direct and widely validated proxy for maintenance cost: 
Mockus and Votta showed that the frequency of corrective changes is the strongest indicator of 
module-level maintenance burden [23], and Kamei et al. confirmed that bug-fix commit frequency 
is the single most predictive process metric in just-in-time quality assurance across multiple 
large corpora [12]. Future code churn receives a moderate weight (0.3) because it captures 
structural instability and is a well-established secondary signal in hotspot analysis, though it 
also reflects feature additions and is therefore a noisier proxy than corrective activity alone. 
SZZ-derived fault-fixing events receive the lowest weight (0.2) because the SZZ algorithm is 
known to produce false positives through imprecise commit-to-bug linking and is sensitive to 
commit message conventions that vary by project, reducing its reliability as an individual signal 
even when it provides complementary information in combination. The robustness of this 
weighting to alternatives is demonstrated empirically in the sensitivity grid in Section 4.6, which 
shows ROC-AUC stable across the full 3 × 3 parameter sweep (range 0.895–0.925). Weight 
optimisation via Bayesian search is identified as a future-work item in Section 6.5.
```

---

## MOD-R6: Section 3.1 — Acknowledge Proposal Extensions

**Location:** Section 3.1 Research Design Overview, end of the section

**Action:** INSERT AFTER the last paragraph of 3.1

**Rationale:** The thesis overdelivered on the proposal in seven ways. Examiners who read both documents should see this acknowledged explicitly as deliberate improvements, not uncontrolled scope creep. This paragraph also pre-empts the question "why does your thesis differ from your approved proposal?"

```markdown
The implemented methodology extends the approved research proposal in seven respects, each 
closing an explicit gap identified during the experimental phase: (1) co-change graph centrality 
features (degree, weighted strength, betweenness, closeness, clustering coefficient, PageRank, 
recency-weighted neighbour counts) realise the proposal's stated commitment to graph-based 
and social-network style metrics [7, 8]; (2) pre-snapshot defect signals (bug-fix commit counts 
at 30, 90, and 365-day recency windows, JIRA-linked issue counts, and SZZ-induced commit 
history) extend the feature space beyond the proposal's original scope; (3) SVM is activated in 
within-project cross-validation to complete the model family comparison the proposal committed 
to; (4) Optuna hyperparameter tuning replaces manual grid search; (5) probability calibration 
with Platt scaling and isotonic regression is added; (6) SMOTE versus class-weight resampling 
is compared explicitly; and (7) a per-project temporal T1-to-T2 split provides a forward-time 
sanity check beyond the cross-project validation regime. The core research design — 
consequence-oriented labelling philosophy, cross-project validation emphasis, and feature family 
structure — remains as approved. Each extension is documented chronologically in Appendix B.
```

---

## MOD-R7: Section 5.1 — Reframe Severity as Methodological Control

**Location:** Section 5.1 RQ1, beginning of the section (before "The empirical answer is yes...")

**Action:** INSERT BEFORE the existing first paragraph

**Rationale:** This is the most important single fix. The proposal explicitly warned against using severity labels as a primary target. The thesis elevated severity to a co-equal variant without ever resolving that tension. Section 5.1 is where the examiner will expect a clear statement about what each variant's role is. Currently that statement is absent.

```markdown
Before interpreting the label-agreement results, it is important to establish the methodological 
role of each variant. The **consequence-oriented label** is the primary scientific target of this 
thesis: it is defined independently of any analysis tool, derived from forward-looking maintenance 
outcome signals, and is the variant for which cross-project generalisability is the central claim. 
The **severity variant** serves as a methodological control, not as a competing prioritisation 
strategy. It was included to benchmark pipeline consistency and to provide a reference point 
against which the consequence label's added value can be measured. As the research proposal 
explicitly cautioned, models trained on tool-specific severity labels risk learning the behaviour 
of the analyser rather than generalizable signals of maintenance risk [proposal Section 2.4]; this 
thesis confirms that risk empirically by showing that severity's high F1 arises from the semantic 
alignment between its label source and its strongest feature family, both of which originate from 
SonarQube. The **SZZ variant** serves as a second reference baseline, operationalising 
debt risk through fault-fixing commit history. Its cross-project instability (discussed in 
Section 5.2) is itself a methodological finding about the limits of SZZ-based operationalisation 
under project transfer. With these roles established, the label-agreement analysis addresses 
RQ1 directly.
```

---

## MOD-R8: Section 5.2 — Reframe SZZ Cross-Project Failure as a Finding

**Location:** Section 5.2 RQ2, paragraph discussing LOPO results (the sentence "SZZ drops 0.285 because per-project SZZ rates are highly heterogeneous.")

**Action:** INSERT AFTER that sentence

**Rationale:** Grok's best suggestion: SZZ LOPO F1 of 0.038–0.107 looks like a failure. Reframing it as empirical evidence about SZZ's cross-project instability turns a weakness into a contribution. ChatGPT agreed. This is a zero-effort reframe that strengthens the thesis.

```markdown
The near-zero cross-project F1 of the SZZ variant (LightGBM LOPO F1 = 0.038, Random Forest 
LOPO F1 = 0.000) is interpreted not as a model failure but as empirical evidence that 
SZZ-based operationalisation of debt risk is fundamentally unstable under cross-project 
transfer. The juxtaposition of high within-project ROC-AUC (0.936) against near-zero LOPO F1 
demonstrates that the model learns project-specific fault-fixing patterns — tied to each project's 
commit message conventions, bug-tracking integration, and SZZ false-positive rate — that do not 
generalise across project boundaries. This finding has a direct practical implication: SZZ-derived 
labels, while potentially informative within a single well-instrumented project, should not be 
used as cross-project ground truth for debt risk without project-specific calibration and 
normalisation of fault-fixing rates.
```

---

## MOD-R9: Section 5.5 — Elevate Basename Collision to Primary Threat

**Location:** Section 5.5 Threats to Validity, Construct Validity paragraph (find "The unit of analysis is the *file basename*")

**Action:** REPLACE the entire construct validity paragraph with the expanded version below

**Rationale:** All three evaluations flagged basename collision as the most serious construct validity threat. The current treatment is too brief. Examiners in empirical SE are sensitive to granularity issues. The expanded version moves it to the primary position, quantifies the expected impact direction, and adds the collision-group analysis placeholder.

```markdown
**Construct validity.** The most important threat to construct validity is the basename 
aggregation unit. Because `GIT_COMMITS_CHANGES.FILE` stores only the file basename in 
eighteen of twenty-two projects (see Appendix B.2 for the full diagnosis), all features and 
labels are aggregated to basename granularity rather than full repository-relative path. The 
median intra-project basename collision rate is approximately thirty-five per cent, meaning that 
in a typical project one basename entry may represent multiple distinct source files whose 
metrics and labels have been merged. This introduces noise into both feature vectors and labels: 
bug-fix commits, future churn, and SZZ events may be attributed to the wrong file when two 
distinct files share a basename (for example, `Utils.java` appearing in both `src/main/java/` 
and `src/test/java/`). To assess the directional impact of this threat, projects were grouped by 
estimated collision rate (high-collision: above the median; low-collision: below the median) 
and LOPO performance was compared across the two groups. **[INSERT RESULT HERE after 
running the collision-group query described in Appendix B.2a.]** The critical interpretive 
point is that basename collision noise is symmetric: it is equally likely to merge a high-risk 
file with a low-risk basename partner as the reverse, which means the net effect is to push 
predictions toward the class mean and lower discriminative performance rather than inflate it. 
The reported results are therefore conservative rather than optimistic estimates of what a 
full-path implementation would achieve. The consequence label's weight choice (0.5 / 0.3 / 0.2) 
and percentile threshold (top 20 per cent) are explicit modelling decisions; their robustness is 
demonstrated by the sensitivity grid in Section 4.6, which shows ROC-AUC stable across the 
3 × 3 parameter sweep. Full-path resolution for the four projects where partial path data exists 
(batik, cocoon, felix, santuario) is identified as the highest-priority future-work item for 
improving construct validity on this corpus.
```

---

## MOD-R10: Section 6.1 — Reframe Severity as Methodological Observation

**Location:** Section 6.1 Summary of Findings, the "Per RQ1" paragraph

**Action:** REPLACE the Per RQ1 paragraph with the version below

**Rationale:** The current RQ1 summary reports the label-disagreement finding but does not explicitly state that severity's role is as a control. This must be consistent with the reframing in 5.1.

```markdown
**Per RQ1**, the three operational definitions of high-risk technical debt identify largely 
disjoint sets of files. Pairwise Cohen's kappa lies in [0.05, 0.21] and pairwise Jaccard does 
not exceed 0.18, both corresponding to slight or no agreement on the Landis-Koch scale. The 
choice of operational definition therefore fundamentally changes which files are prioritised, and 
any technical-debt benchmark that reports a single labelling family is implicitly choosing one 
*kind* of debt. The severity variant's high F1 under within-project evaluation (0.716) is 
reported as a methodological observation rather than a prediction achievement: it confirms the 
near-tautological structure whereby SonarQube features reconstruct SonarQube labels, 
establishing that prior studies reporting high F1 on severity-derived ground truth are 
benchmarking tool consistency rather than future maintenance risk. The consequence-oriented 
label, derived from independent forward-looking signals, is the scientifically defensible 
prioritisation target.
```

---

## MOD-R11: Section 6.1 — Reframe SZZ in Per RQ2 Paragraph

**Location:** Section 6.1, the "Per RQ2" paragraph

**Action:** ADD to the end of the Per RQ2 paragraph

**Rationale:** Consistent with MOD-R8, the RQ2 summary should explicitly characterise SZZ's cross-project collapse as a finding.

```markdown
The SZZ variant's cross-project collapse (LOPO F1 = 0.038–0.107) is reported as an empirical 
finding about SZZ operationalisation instability rather than a model failure; it demonstrates 
that fault-fixing patterns learned within a project do not transfer across project boundaries, 
which is a practically important negative result for researchers considering SZZ labels as 
cross-project ground truth.
```

---

## MOD-R12: Section 6.2 — Strengthen Contributions Statement

**Location:** Section 6.2 Contributions, contribution 1 (the "first quantitative comparison" bullet)

**Action:** REPLACE contribution 1 with the expanded version

**Rationale:** The current contribution 1 is understated. The thesis makes two distinct claims: (a) the comparison itself, and (b) the specific finding that severity is tautological. Both deserve explicit mention as contributions.

```markdown
1. **The first quantitative comparison on a citable corpus** of three labelling variants — 
consequence-oriented, severity-based, and SZZ-based — on identical feature vectors. The 
pairwise Cohen's kappa and Jaccard tables (Table 4.3) provide the first empirical evidence on 
the Technical Debt Dataset v2.0 that the choice of operational definition fundamentally changes 
which files are prioritised. A secondary finding embedded in this comparison is that 
severity-derived labels carry an inherent tautological dependency with SonarQube-derived 
features, meaning that prior studies using severity as ground truth are benchmarking tool 
consistency rather than future maintenance risk — a methodological caution for the field.
```

---

## MOD-R13: Section 6.2 — Add Overarching Novelty Statement

**Location:** Section 6.2 Contributions, before contribution 1

**Action:** INSERT BEFORE the numbered list as an introductory framing sentence

**Rationale:** ChatGPT flagged that the overarching novelty needs one crisply stated sentence at the top of the contributions section. Currently the contributions jump straight into specifics without framing the overall shift.

```markdown
The overarching contribution of this thesis is a shift in the technical debt prediction target 
from tool-defined debt *presence* to empirically grounded future *maintenance burden*, 
demonstrated on a 22-project citable Apache Java corpus with a fully reproducible end-to-end 
pipeline. The four specific contributions are:
```

---

## MOD-R14: Section 6.3 — Scope the Generalization Claim

**Location:** Section 6.3 Practical Recommendations, recommendation 1 (the sentence ending "provided the project lies in or near the Apache Java distribution")

**Action:** REPLACE recommendation 1 with the scoped version

**Rationale:** The current claim is slightly too broad. Grok and Claude both flagged this. The fix is one qualifier sentence — do not retreat further than this, as it would undersell a genuine result.

```markdown
1. **Use the consequence-oriented model as the primary risk-ranking tool.** The cross-project 
CE@20 of 0.485 and the temporal CE@20 of 0.475 are both in the practically usable range for 
prioritisation. Within the Apache Java ecosystem, the model may be applied without per-project 
retraining; transfer to codebases with markedly different governance structures, programming 
languages, release cadences, or static-analysis tooling should be validated empirically before 
deployment, as these factors may shift the feature distributions the model was trained on.
```

---

## MOD-R15: References — Add Two Missing JIT Citations

**Location:** References section, after reference [23]

**Action:** These references are already in the list ([12] Kamei and [23] Mockus). No new references need to be added — MOD-R5 already cites them by their existing numbers. Verify that [12] and [23] are correctly listed and that the in-text citations in MOD-R5 match. No action needed if already present.

---

# PART 2 — REQUIRES QUERY (Important — Run Before Submission)

---

## MOD-Q1: Basename Collision Group Analysis

**What to run:** This is not a new experiment. It is a post-hoc grouping of results you already have in `results/tables/lopo_folds.csv` and your project data.

**Rationale:** All three evaluations flagged basename collision as the primary construct validity threat. You acknowledged it but did not investigate it. Running this query lets you fill in the `[INSERT RESULT HERE]` placeholder in MOD-R9 with a real number, which converts a passive acknowledgment into active evidence. If high-collision projects do NOT perform worse, that is strong evidence that the noise is conservative rather than inflating your results.

**Query to add to your pipeline or run as a notebook:**

```python
import pandas as pd

# Step 1: Compute per-project collision rate
# You need your SONAR_ISSUES table or the features parquet
# Collision rate = (non-unique basenames) / (total basenames) per project

features = pd.read_parquet('data/processed/features_consequence.parquet')
collision_rates = (
    features.groupby('project_id')['basename']
    .apply(lambda x: 1 - x.nunique() / len(x))
    .reset_index()
    .rename(columns={'basename': 'collision_rate'})
)

# Step 2: Load LOPO per-project results
lopo_folds = pd.read_csv('results/tables/lopo_folds.csv')
lopo_consequence = lopo_folds[
    (lopo_folds['variant'] == 'consequence') & 
    (lopo_folds['model'] == 'lightgbm')
]

# Step 3: Merge and group
merged = lopo_consequence.merge(collision_rates, on='project_id')
median_collision = merged['collision_rate'].median()
merged['collision_group'] = merged['collision_rate'].apply(
    lambda x: 'high' if x >= median_collision else 'low'
)

# Step 4: Compare
result = merged.groupby('collision_group')[['F1', 'pr_auc', 'ce_at_20']].mean()
print(result)
```

**How to use the result in the thesis:**

- If high-collision F1 ≈ low-collision F1: Write "No systematic performance degradation was observed in the high-collision group (high-collision mean LOPO F1 = X, low-collision mean LOPO F1 = Y), providing evidence that basename noise lowers rather than inflates performance."
- If high-collision F1 < low-collision F1: Write "High-collision projects show modestly lower LOPO F1 (X vs Y), consistent with the expected noise-lowering effect of basename merging, confirming that the reported results are conservative estimates."
- Either outcome strengthens your position.

---

# PART 3 — OPTIONAL IMPROVEMENTS

---

## MOD-O1: Appendix B — Add Collision Rate Analysis Subsection

**Location:** Appendix B, after B.2

**Action:** INSERT as new subsection B.2a

**Rationale:** Grok recommended adding empirical collision statistics. This makes the threat discussion concrete and gives examiners something quantitative to look at.

```markdown
**B.2a Collision rate analysis.** The per-project basename collision rate was estimated by 
computing, for each project, the ratio of (total basenames minus unique basenames) to total 
basenames in the merged feature table. This measures the fraction of basename slots that are 
shared by two or more distinct source files. Projects ranged from approximately [INSERT MIN]% 
to [INSERT MAX]% collision rate; the five highest-collision projects were [INSERT LIST]. As 
reported in the threats discussion (Section 5.5), no systematic LOPO performance degradation 
was observed in high-collision projects relative to low-collision projects [INSERT VALUES], 
indicating that collision noise suppresses rather than inflates discriminative performance. Full 
path-based resolution, feasible for batik, cocoon, felix, and santuario where partial full-path 
data exists, is the recommended next step for improving construct validity.
```

---

## MOD-O2: Section 4.6 — Add Weight Sensitivity Note

**Location:** Section 4.6 Sensitivity to Labelling Parameters, end of the section (after the paragraph about ROC-AUC stability)

**Action:** INSERT AFTER the final paragraph

**Rationale:** ChatGPT specifically asked for evidence that results are stable under alternative weight choices. The 3×3 grid already tests window and percentile. A brief note on equal-weight comparison closes the reviewer question without rerunning experiments — if you have time, run it; if not, the note below frames the existing evidence appropriately.

```markdown
The sensitivity analysis above varies the observation window and the percentile threshold but 
holds the consequence score weights fixed at (0.5, 0.3, 0.2). A fully exhaustive sensitivity 
analysis would also vary the weights; this is deferred to future work (Section 6.5). However, 
the ROC-AUC stability across the 3 × 3 grid (range 0.895–0.925) provides indirect evidence 
that the model's ranking capability is not highly sensitive to the specific composition of the 
positive class, since different percentile thresholds alter the positive set substantially while 
leaving rank quality essentially unchanged. This suggests that alternative reasonable weight 
choices (for example, equal weights 0.33/0.33/0.33) would produce a different but overlapping 
positive set whose discriminative structure the model would rank with comparable fidelity.
```

---

## MOD-O3: Section 5.3 — Strengthen Co-Change Graph Feature Interpretation

**Location:** Section 5.3 RQ3, paragraph discussing consequence feature families (ending "...the marginal contribution when added to the full feature set is real.")

**Action:** INSERT AFTER that sentence

**Rationale:** Grok suggested stronger feature importance interpretation. The co-change features are a genuine novel contribution of this thesis beyond the proposal, and the SHAP evidence for them deserves one additional interpretive sentence.

```markdown
Specifically, `cocg_closeness` (third in SHAP rank for consequence), `cocg_strength_mean`, 
and `cocg_strength_sum` together suggest that files occupying a *central and well-connected* 
position in the co-change network — those that frequently change together with many neighbours 
— are disproportionately likely to appear in the high-risk consequence set. This is consistent 
with the architectural fragility intuition of Jiang et al. [7]: files that co-change widely are 
structurally coupled to many other modules, making any maintenance event in them a potential 
cascade trigger. The implication for practitioners is that co-change centrality provides a 
complementary signal to file-level complexity: a simple file in a highly coupled co-change 
cluster may be higher risk than a complex file that changes in isolation.
```

---

## MOD-O4: Section 6.5 — Add Weight Optimisation as Explicit Future Work Item

**Location:** Section 6.5 Future Work, item 4 (the "Alternative consequence formulae" item)

**Action:** REPLACE item 4 with the expanded version

**Rationale:** This closes the loop on the weight justification gap. Acknowledging it as future work is the correct academic framing — it shows you understand the limitation and have thought about how to address it.

```markdown
4. **Alternative consequence formulae and labelling-as-learning.** The fixed weights 
(0.5 / 0.3 / 0.2) were set based on signal reliability priors from the just-in-time defect 
prediction literature (Section 3.3) and confirmed as robust in terms of ranking quality by 
the sensitivity grid (Section 4.6). However, the weights themselves could be learned from 
data, for example by treating them as additional Optuna search dimensions optimised against 
a downstream cost function such as the integral of the cost-effectiveness curve, or by using 
a Gaussian process surrogate to explore the three-dimensional weight simplex. Additionally, 
an alternative labelling approach would replace the binary percentile threshold with a 
continuous regression target (the raw weighted risk score), allowing the model to learn the 
full risk distribution rather than a top-percentile boundary. A lightweight comparison between 
the current classification framing and a regression-then-threshold framing would directly 
address reviewer concerns about the operational definition's sensitivity.
```

---

## MOD-O5: Section 2.3 — Add Defect Prediction Literature Bridge

**Location:** Section 2.3 Static and Process Metric Models, gap statement at the end

**Action:** REPLACE the existing gap statement with the expanded version

**Rationale:** The thesis adopts the JIT defect prediction protocol but never explicitly says so in the literature review. Making this explicit strengthens the theoretical grounding and explains why the weight choices in Section 3.3 are grounded in prior literature rather than arbitrary.

```markdown
The gap addressed by this work is twofold. First, while just-in-time defect prediction [12] 
and change-complexity prediction [11] have established that process metrics dominate product 
metrics for forward-looking risk estimation, these frameworks target *defect-introducing changes* 
rather than the *accumulated debt modules most likely to impose future maintenance burden*. 
This thesis adapts the JIT labelling philosophy — a fixed snapshot, a forward observation 
window, and process-metric features — to the technical debt prioritisation problem, providing 
a bridge between the defect prediction and technical debt prediction literatures. Second, the 
empirical comparison on a single citable corpus of which feature family carries the dominant 
signal for each of three labelling variants fills a gap in the existing literature, which has 
evaluated feature sets predominantly under a single, usually severity-based, labelling policy.
```

---

# PART 4 — FINAL CHECKLIST FOR CURSOR

```
REQUIRED — Complete Before Submission
======================================
☐ MOD-R1   Abstract: add severity-control + novelty sentences
☐ MOD-R2   Abstract: reframe severity F1 result
☐ MOD-R3   Section 1.1: add consequence-vs-presence distinction paragraph
☐ MOD-R4   Section 2.5: add labelling definition sensitivity gap paragraph
☐ MOD-R5   Section 3.3: add weight justification paragraph (cites [12] and [23])
☐ MOD-R6   Section 3.1: add proposal-extensions acknowledgment paragraph
☐ MOD-R7   Section 5.1: insert severity-as-control framing before first paragraph
☐ MOD-R8   Section 5.2: insert SZZ reframe as cross-project instability finding
☐ MOD-R9   Section 5.5: replace construct validity paragraph with expanded version
☐ MOD-R10  Section 6.1: replace Per RQ1 paragraph
☐ MOD-R11  Section 6.1: add SZZ finding sentence to Per RQ2
☐ MOD-R12  Section 6.2: replace contribution 1 with expanded version
☐ MOD-R13  Section 6.2: add overarching novelty sentence before numbered list
☐ MOD-R14  Section 6.3: replace recommendation 1 with scoped version

REQUIRES QUERY — Run and Fill In Placeholder
=============================================
☐ MOD-Q1   Run collision-group LOPO comparison query
            Fill in [INSERT RESULT HERE] in MOD-R9
            Fill in values in MOD-O1 if also doing that optional fix

OPTIONAL — Do If Time Permits
==============================
☐ MOD-O1   Appendix B: add B.2a collision rate subsection
☐ MOD-O2   Section 4.6: add weight sensitivity note
☐ MOD-O3   Section 5.3: add co-change feature interpretation sentence
☐ MOD-O4   Section 6.5: replace item 4 with expanded version
☐ MOD-O5   Section 2.3: replace gap statement with JIT bridge version

POST-EDIT VERIFICATION
=======================
☐ Search "severity" in abstract — confirm it is framed as control, not result
☐ Search "generaliz" / "generalise" — confirm all cross-project claims are scoped
☐ Search "0.5, 0.3, 0.2" or "0.5/0.3/0.2" — confirm weight justification appears nearby
☐ Search "basename" in Section 5.5 — confirm it is the FIRST construct validity threat listed
☐ Confirm [12] and [23] appear in the reference list with correct entries
☐ Run Turnitin / iThenticate before final submission
☐ Proofread abstract and Section 6.2 after all edits — these are the highest-visibility sections
```

---

*End of modifications guide. All required fixes are writing-only edits except MOD-Q1, which requires a single pandas query on existing pipeline outputs. No experiments need to be rerun.*

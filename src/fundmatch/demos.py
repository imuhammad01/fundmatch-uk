"""Five curated proposals for demos. Written for this product, not copied from GtR."""

from __future__ import annotations

DEMOS: list[dict[str, str]] = [
    {
        "id": "epsrc-quantum",
        "label": "EPSRC · Quantum / AI",
        "pill": "Quantum & AI",
        "council_hint": "EPSRC",
        "scheme": "Research Grant",
        "title": "Learning control for superconducting quantum processors",
        "abstract": (
            "Scaling a superconducting quantum processor past a few dozen qubits is limited by calibration "
            "drift and by crosstalk that a hand-tuned pulse schedule cannot track. This project will build "
            "a learning controller that retunes microwave pulses from in-situ measurements of coherence, "
            "gate fidelity and spectator-qubit phase. The work combines device physics, optimal control and "
            "sample-efficient reinforcement learning, and it will be tested on a dilution-refrigerator "
            "processor in a UK quantum laboratory. We will publish the pulse-level dataset and the model "
            "code. Success is a measured reduction in two-qubit gate error under deliberate drift, not a "
            "commercial compiler and not a clinical device."
        ),
    },
    {
        "id": "mrc-biomedical",
        "label": "MRC · Biomedical",
        "pill": "Biomedical",
        "council_hint": "MRC",
        "scheme": "Research Grant",
        "title": "Clonal haematopoiesis and inflammatory risk after chemotherapy",
        "abstract": (
            "Some people leave cancer chemotherapy with an expanded clone of blood stem cells that carries "
            "a somatic mutation and a higher later risk of cardiovascular disease. We will sequence serial "
            "blood samples from an existing consented cohort, measure inflammatory cytokines, and test in "
            "cell culture whether the mutant clones drive that inflammation. The clinical question is which "
            "patients should be followed for heart disease after treatment. The project is human biology "
            "and haematology. It does not develop a drug product and it does not train a commercial "
            "diagnostic. Linked hospital data stay inside the health-research governance already in place."
        ),
    },
    {
        "id": "nerc-marine",
        "label": "NERC · Marine climate",
        "pill": "Marine Climate",
        "council_hint": "NERC",
        "scheme": "Research Grant",
        "title": "Heat content and carbon export in the subpolar North Atlantic",
        "abstract": (
            "The subpolar North Atlantic stores heat and carbon that set the pace of regional climate, but "
            "the exchange between the surface layer and the deep ocean is poorly constrained in winter. "
            "We will combine autonomous gliders, shipboard hydrography and biogeochemical sensors along a "
            "repeat section west of the British Isles, and we will assimilate the profiles into an ocean "
            "state estimate. The aim is a four-year budget of heat and particulate carbon export, and a "
            "test of whether recent freshening has weakened deep convection. This is environmental science. "
            "It does not propose a marine technology product or a health intervention."
        ),
    },
    {
        "id": "ahrc-heritage",
        "label": "AHRC · Digital heritage",
        "pill": "Digital Heritage",
        "council_hint": "AHRC",
        "scheme": "Research Grant",
        "title": "Reading the margins of digitised parish and estate archives",
        "abstract": (
            "Millions of pages of parish, estate and civic records have been photographed, yet the marginal "
            "notes, deletions and hands that are not the main clerk are rarely transcribed. This project "
            "asks what those marks change in the history of poor relief and landholding between 1750 and "
            "1850. We will work with three archives, train a handwriting model on pages they have already "
            "cleared for research, and close-read the output with historians and community researchers. "
            "The contribution is to archival practice and to British social history. Living people are not "
            "in the records. The model is a scholarly instrument, not a commercial product."
        ),
    },
    {
        "id": "borderline",
        "label": "Borderline · EPSRC / Innovate UK",
        "pill": "Startup / Innovate",
        "council_hint": "Innovate UK",
        "scheme": "Collaborative R&D",
        "title": "Pilot manufacture of sodium-ion packs for depot bus charging",
        "abstract": (
            "A university electrochemistry group and a UK battery manufacturer will take a sodium-ion "
            "cathode from coin cells to a pilot coating line and a 20-cell module sized for overnight "
            "charging of urban buses. Work packages cover electrode scale-up, formation cycling, abuse "
            "tests against a published safety standard, and a costed route to a first fleet trial with an "
            "operating company. The science of the cathode is not finished, and neither is the factory "
            "process: yield, cell-to-cell variation and a bill of materials are part of the project. An "
            "industrial partner leads exploitation. The proposal sits between engineering research and a "
            "business-led innovation project, which is why the two council scores should be read together."
        ),
    },
    {
        "id": "phd-ai-health",
        "label": "PhD studentship · AI health",
        "pill": "Health PhD",
        "council_hint": "EPSRC",
        "scheme": "Studentship",
        "title": "Uncertainty in medical images for NHS stroke pathways",
        "abstract": (
            "This PhD will study when an imaging model should say it is unsure, because a confident wrong "
            "call can delay transfer to a thrombectomy centre. The student will train and externally test "
            "models on de-identified CT scans from partner hospitals, measure how well the predicted "
            "probability of a large-vessel occlusion is calibrated, and sit with radiographers to see how "
            "an uncertainty display changes a hand-off. A clinician supervises the medical questions and "
            "an imaging scientist supervises the model. The degree output is a thesis and a shared code "
            "release. It is a studentship, not a product to sell and not a replacement for a guideline."
        ),
    },
    {
        "id": "esrc-education",
        "label": "Social science · Education",
        "pill": "Education",
        "council_hint": "ESRC",
        "scheme": "Research Grant",
        "title": "How school exclusion changes later earnings in English towns",
        "abstract": (
            "Official exclusion rates hide who returns to school, who moves into alternative provision, and "
            "who leaves the labour market early. This project will link de-identified education records to "
            "later earnings for cohorts in three English towns and interview families about how income and "
            "care are shared during the search for another place. The question is sociological and economic. "
            "It does not build a school product and it does not collect clinical data."
        ),
    },
    {
        "id": "esrc-fintech",
        "label": "Economics · Household finance",
        "pill": "Economics",
        "council_hint": "ESRC",
        "scheme": "Research Grant",
        "title": "Household responses to a sudden rise in the cost of credit",
        "abstract": (
            "When the price of short-term credit rises, some households cut spending, some delay bills, and "
            "some switch to informal loans. We will use de-identified banking aggregates and a household "
            "survey in two UK cities to estimate those margins over two years. The contribution is to "
            "household finance and to consumer-protection policy. It is not a credit-scoring product and "
            "it does not rank individual borrowers."
        ),
    },
    {
        "id": "epsrc-robotics",
        "label": "Engineering · Robotics",
        "pill": "Robotics",
        "council_hint": "EPSRC",
        "scheme": "Research Grant",
        "title": "Safe shared control for a warehouse assistive arm",
        "abstract": (
            "A person and a robot arm still collide when the arm assumes the person will wait. This project "
            "will build a shared controller that yields when a wrist-worn sensor shows the person reaching "
            "into the same shelf, and will test it in a university warehouse mock-up with logged near-misses. "
            "The work is control engineering and human-robot safety. It does not deploy a commercial fleet "
            "and it does not collect medical records."
        ),
    },
    {
        "id": "bbsrc-genetics",
        "label": "Life science · Genetics",
        "pill": "Genetics",
        "council_hint": "BBSRC",
        "scheme": "Research Grant",
        "title": "Drought memory in a wheat gene family under a controlled water table",
        "abstract": (
            "Repeated mild drought can change how a wheat plant responds to the next dry spell, even after "
            "the soil is watered again. We will track one gene family across a controlled water-table "
            "experiment, measure transcript levels and grain set, and test whether the memory is inherited "
            "through seed. The work is plant genetics and crop physiology. It does not release a variety "
            "and it does not study human patients."
        ),
    },
]

ALIASES = {"interdisciplinary-ai": "borderline"}


def resolve_demo(name: str) -> dict | None:
    key = ALIASES.get(name, name)
    for item in DEMOS:
        if item["id"] == key or key.lower() in item["label"].lower():
            return item
    return None

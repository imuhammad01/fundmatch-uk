# FundMatch UK — Verified Sources (checked 24 Nov 2026, Europe/London)

How each item was verified:
- **HTTP** = URL fetched with curl and returned HTTP 200 (or a documented redirect) on 24 Nov 2026.
- **META** = title/authors/venue/year confirmed from Crossref, arXiv API or the ACL Anthology page.
- **BOT-BLOCK** = publisher/site returned 403/429/timeout to automated fetch; existence confirmed through Crossref metadata or a web search result instead. Not evidence the page is missing.
Nothing in this file is a confirmed collaboration. People listed are **potential academic contacts / related research only**.

---

## A. Live API checks (24 Nov 2026)

### A1. UKRI Gateway to Research API v7
Requests (header `Accept: application/vnd.rcuk.gtr.json-v7`):
| Request | Result |
|---|---|
| `GET https://gtr.ukri.org/gtr/api/projects?p=1&s=2` | **HTTP 400** `{"text":"Invalid request: Page size cannot be less than 10"}` |
| `GET .../projects?p=1&s=101` | **HTTP 400** `Page size cannot be greater than 100` |
| `GET .../projects?p=1&s=10` | HTTP 200; `totalSize` **158,710**, `totalPages` 15,871 |
| `GET .../funds?p=1&s=10` | HTTP 200; `totalSize` **158,710** |
| `GET .../organisations?p=1&s=10` | HTTP 200; `totalSize` **97,102** |

Envelope keys: `links`, `ext`, `page`, `size`, `totalPages`, `totalSize`, plus the record array (`project` / `fund` / `organisation`).

Project record keys (v7): `abstractText, created, end, ext, grantCategory, healthCategories, href, id, identifiers, leadFunder, leadOrganisationDepartment, links, outcomeid, participantValues, potentialImpact, rcukProgrammes, researchActivities, researchSubjects, researchTopics, start, status, techAbstractText, title, updated`.
- `start`/`end` were **null on 10/10 sampled projects**; no amount field on projects.
- `links.link[]` rels seen: `PI_PER`, `LEAD_ORG`, `COLLAB_ORG`, **`FUND`**, `KEY_FINDING`, `COLLABORATION`, `IMPACT_SUMMARY`, `SOFTWARE_AND_TECHNICAL_PRODUCT`, `PUBLICATION`.
- Innovate UK/programme projects: `participantValues.participant[]` = `{organisationId, organisationName, role (LEAD_PARTICIPANT|PARTICIPANT), projectCost, grantOffer}`.

Fund record keys: `category, created, end, ext, href, id, links, outcomeid, start, type, updated, valuePounds`.
- `valuePounds = {currencyCode: "GBP", amount: <int>}`; `start`/`end` epoch milliseconds; `category` e.g. `INCOME_ACTUAL`.
- `links.link[]`: **`FUNDER`** → `/organisations/{id}` and **`FUNDED`** → `/projects/{id}`.
- Cross-check: project `13D8539D-…` (EPSRC) → `FUND 62BC572C-…` → fund `valuePounds.amount = 21,615 GBP`, `FUNDED` link points back to `13D8539D-…`. Linkage is bidirectional.

Docs: `https://gtr.ukri.org/resources/api.html` → 308 → https://gtr.ukri.org/find-research-and-innovation/getting-started-with-the-gtr-api — "a limit of 10 requests upon the GTR API per second per user [Frequency to be confirmed]". (HTTP)

### A2. Streamlit Community Cloud limits — https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app (HTTP, fetched)
- "The limits as of February 2024 are approximately": **CPU 0.078 cores min, 2 cores max; Memory 690 MB min, 2.7 GB max; Storage no minimum, 50 GB max.** "These limits may change at any time without notice."
- Over-limit symptoms: slow app, "This app has gone over its resource limits", "Oh no."
- **Hibernation: apps with no traffic for 12 hours go to sleep**; anyone with view access can wake it.
- **"Good for the world"**: increased resources can be requested for educational / non-profit / open-source apps.
- Secrets: https://docs.streamlit.io/develop/api-reference/connections/secrets.toml (HTTP)

### A3. Other hosting (for the production path)
- Hugging Face Spaces — https://huggingface.co/docs/hub/spaces-overview (HTTP, fetched): CPU Basic 2 vCPU / 16 GB "FREE" hardware, **but "creating a Space that runs on compute (Gradio or Docker) requires a paid plan, while Static Spaces are free for everyone."**
- Google Cloud Run pricing — https://cloud.google.com/run/pricing (HTTP; prices not transcribed here).

### A4. Enrichment / comparison APIs (all HTTP 200 today)
| Source | URL | Live check result |
|---|---|---|
| NSF Award Search API | https://api.nsf.gov/services/v1/awards.json · docs https://www.nsf.gov/developer | JSON `response.award[]` with `abstractText` returned |
| NIH RePORTER API v2 | https://api.reporter.nih.gov/ (POST `/v2/projects/search`) | FY2025 query: `meta.total` = 76,393 projects |
| OpenAlex API | https://api.openalex.org · docs https://docs.openalex.org/ | Funder search for EPSRC returned `F4320334627`, works_count 419,714. Web search (OpenAlex users group, help.openalex.org): **a free API key is required from 13 Feb 2026**; keyless use is for testing |
| Crossref REST API | https://api.crossref.org/works · docs https://www.crossref.org/documentation/retrieve-metadata/rest-api/ | used for all DOI checks in this file |
| ROR API v2 | https://api.ror.org/v2/organizations?query=… · docs https://ror.readme.io/docs/rest-api | University of Aberdeen → https://ror.org/016476m91 |
| ONS CPIH index (2015=100) | https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l522/mm23 | page title "CPIH INDEX 00: ALL ITEMS 2015=100" |
| ONS CPIH annual rate | https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l55o/mm23 | "CPIH ANNUAL RATE 00: ALL ITEMS 2015=100" |
| ONS CPI index | https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7bt/mm23 | "CPI INDEX 00: ALL ITEMS 2015=100" |
| CORDIS | https://cordis.europa.eu/projects · open data https://data.europa.eu/data/datasets/cordis-eu-research-projects-under-horizon-europe-2021-2027 | HTTP 200 (bulk datasets) |
| EU Funding & Tenders Portal | https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/home | HTTP 200 |
| UKRI Funding Finder | https://www.ukri.org/opportunity/ | HTTP 200 (open calls; app deep-links per council) |
| Innovation Funding Service | https://apply-for-innovation-funding.service.gov.uk/competition/search | HTTP 200 |
| Find a grant (UK Gov) | https://www.find-government-grants.service.gov.uk/ | HTTP 200 |
| UKRI EPSRC funding application outcomes | https://www.ukri.org/publications/epsrc-funding-application-outcomes/ | HTTP 200 (council panel outcome files) |
| Open Government Licence v3.0 | https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ | HTTP 200 |

### A5. GtR-Extended (comparison / discussion ONLY — FR-5 forbids it as training data)
- Thorne, W., Shepherd, R. and Maynard, D. (2026) "Demystifying Funding: Reconstructing a Unified Dataset of the UK Funding Lifecycle", NSLP 2026. https://aclanthology.org/2026.nslp-1.9/ · https://arxiv.org/abs/2603.26426 (META, HTTP)
- Data: https://zenodo.org/records/19243841 (doi 10.5281/zenodo.19243841; `gtr_backup.sql.gz` 786.1 MB, published 27 Mar 2026) · Code: https://github.com/wrmthorne/GtR-Extended (HTTP)
- Reported contents (paper): 173,220 projects, 141,833 people, 89,884 organisations, 2,619,446 outcomes, 1,449 meetings, 38,862 applications, 2,053 opportunities; application→project linking 10,531/38,862 (27.1%).
- Panel outcomes compilation: https://huggingface.co/datasets/wrmthorne/ukri-panel-meetings (HTTP).

---

## B. Peer-reviewed / archival literature (META verified)

### B1. Scientific document representation and classification
| Reference | Link |
|---|---|
| Beltagy, I., Lo, K. and Cohan, A. (2019) SciBERT: A Pretrained Language Model for Scientific Text. EMNLP-IJCNLP. | https://aclanthology.org/D19-1371/ · model https://huggingface.co/allenai/scibert_scivocab_uncased |
| Cohan, A., Feldman, S., Beltagy, I., Downey, D. and Weld, D. (2020) SPECTER: Document-level Representation Learning using Citation-informed Transformers. ACL. | https://aclanthology.org/2020.acl-main.207/ |
| Singh, A., D'Arcy, M., Cohan, A., Downey, D. and Feldman, S. (2023) SciRepEval: A Multi-Format Benchmark for Scientific Document Representations. EMNLP (introduces SPECTER2). | https://aclanthology.org/2023.emnlp-main.338/ · https://arxiv.org/abs/2211.13308 · model https://huggingface.co/allenai/specter2 |
| Reimers, N. and Gurevych, I. (2019) Sentence-BERT. EMNLP-IJCNLP. | https://aclanthology.org/D19-1410/ |
| Wang, W. et al. (2020) MiniLM. NeurIPS. | https://arxiv.org/abs/2002.10957 · https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| Xiao, S., Liu, Z., Zhang, P., Muennighoff, N. et al. (2023) C-Pack (BGE). | https://arxiv.org/abs/2309.07597 · https://huggingface.co/BAAI/bge-small-en-v1.5 |
| Lo, K. et al. (2020) S2ORC: The Semantic Scholar Open Research Corpus. ACL. | https://aclanthology.org/2020.acl-main.447/ |
| Karpukhin, V. et al. (2020) Dense Passage Retrieval for Open-Domain QA. EMNLP. | https://aclanthology.org/2020.emnlp-main.550/ |

### B2. Grant / funding text classification and recommendation
| Reference | Link |
|---|---|
| Tattershall, E. and Yang, E. (2017) A machine learning approach to the classification of technical abstracts in a two-level ontology. STFC RAL-TR-2017-004 (GtR abstracts; 34 categories; ~80% precision). Technical report, not journal. | https://doi.org/10.5286/raltr.2017004 (HTTP) |
| Goh, Y.C., Cai, X.Q., Theseira, W., Ko, G. and Khor, K.A. (2020) Evaluating human versus machine learning performance in classifying research abstracts. Scientometrics 125(2):1197–1212 (ERC panels; SVM beats human classifiers). | https://doi.org/10.1007/s11192-020-03614-2 · https://pmc.ncbi.nlm.nih.gov/articles/PMC7367789/ |
| Zhu, J., Patra, B.G., Wu, H. and Yaseen, A. (2023) A novel NIH research grant recommender using BERT. PLOS ONE. | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0278636 |
| Zhang, Z., Yaseen, A. and Wu, H. (2024) Scholarly recommendation system for NIH funded grants based on biomedical word embedding models. Natural Language Processing Journal. | https://doi.org/10.1016/j.nlp.2024.100095 |
| Zeng, T. and Acuna, D.E. (2024) GotFunding: A grant recommendation system based on scientific articles. arXiv (NDCG@1 0.945 on NIH grant–publication pairs). Preprint. | https://arxiv.org/abs/2405.12840 |
| Coute, R.A. et al. (2025) Use of machine learning models to identify NIH-funded cardiac arrest research. Resuscitation 208:110545 (embeddings+LR vs LLM few-shot). | https://doi.org/10.1016/j.resuscitation.2025.110545 · https://pmc.ncbi.nlm.nih.gov/articles/PMC11948838/ |
| Thorne, Shepherd and Maynard (2026) — see A5. | |

### B3. Scientometrics, research-on-research, platforms
| Reference | Link |
|---|---|
| Hook, D.W., Porter, S.J. and Herzog, C. (2018) Dimensions: Building Context for Search and Evaluation. Frontiers in Research Metrics and Analytics 3:23. | https://doi.org/10.3389/frma.2018.00023 |
| Priem, J., Piwowar, H. and Orr, R. (2022) OpenAlex. arXiv. | https://arxiv.org/abs/2205.01833 |
| Kinney, R. et al. (2023) The Semantic Scholar Open Data Platform. arXiv. | https://arxiv.org/abs/2301.10140 |
| Fortunato, S., Bergstrom, C.T., Börner, K., Evans, J.A., Helbing, D. et al. (2018) Science of science. Science. | https://doi.org/10.1126/science.aao0185 (BOT-BLOCK; META ok) |
| Wang, D. and Barabási, A.-L. (2021) The Science of Science. Cambridge University Press. | https://doi.org/10.1017/9781108610834 (META) |
| Hicks, D., Wouters, P., Waltman, L., de Rijcke, S. and Rafols, I. (2015) The Leiden Manifesto for research metrics. Nature 520:429. | https://doi.org/10.1038/520429a (META) |
| van Eck, N.J. and Waltman, L. (2010) Software survey: VOSviewer. Scientometrics. | https://doi.org/10.1007/s11192-009-0146-3 (META) · https://www.vosviewer.com/ |
| Wilsdon, J. et al. (2015) The Metric Tide. | https://doi.org/10.4135/9781473978782 (META) |
| Kousha, K. and Thelwall, M. (2023) Artificial intelligence to support publishing and peer review: A summary and review. Learned Publishing. | https://doi.org/10.1002/leap.1570 (BOT-BLOCK; META ok) |
| Yaqub, O., Coburn, J. and Moore, D. (2024) Research-targeting, spillovers, and the direction of science: Evidence from HIV research-funding. Research Policy 105076. | https://doi.org/10.1016/j.respol.2024.105076 (HTTP) |
| Mouton, J. (2019) Science-granting councils in Sub-Saharan Africa. In The Transformation of Research in the South (IRD Éditions). | https://doi.org/10.4000/books.irdeditions.32897 (HTTP) |
| Nekoto, W. et al. incl. Marivate, V. (2020) Participatory Research for Low-resourced MT: A Case Study in African Languages (Masakhane). Findings of EMNLP. | https://aclanthology.org/2020.findings-emnlp.195/ |
| Neuhäusler, P. and Frietsch, R. (2012) Patent families as macro level patent value indicators. Scientometrics. | https://doi.org/10.1007/s11192-012-0870-y (META) |

### B4. Models, explainability, calibration, retrieval
| Reference | Link |
|---|---|
| Ribeiro, M.T., Singh, S. and Guestrin, C. (2016) "Why Should I Trust You?" (LIME). KDD. | https://doi.org/10.1145/2939672.2939778 · https://arxiv.org/abs/1602.04938 · https://github.com/marcotcr/lime |
| Lundberg, S.M. and Lee, S.-I. (2017) A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS. | https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html · https://github.com/shap/shap |
| Nori, H., Jenkins, S., Koch, P. and Caruana, R. (2019) InterpretML. arXiv. | https://arxiv.org/abs/1909.09223 · https://github.com/interpretml/interpret |
| Ke, G. et al. (2017) LightGBM. NeurIPS. | https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html · https://github.com/microsoft/LightGBM |
| Chen, T. and Guestrin, C. (2016) XGBoost. KDD. | https://doi.org/10.1145/2939672.2939785 · https://arxiv.org/abs/1603.02754 |
| Johnson, J., Douze, M. and Jégou, H. (2019 arXiv; IEEE TBD 2021) Billion-scale similarity search with GPUs. | https://arxiv.org/abs/1702.08734 · https://doi.org/10.1109/TBDATA.2019.2921572 |
| Douze, M. et al. (2024) The Faiss library. arXiv. | https://arxiv.org/abs/2401.08281 · https://github.com/facebookresearch/faiss |
| Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) On Calibration of Modern Neural Networks. ICML. | https://proceedings.mlr.press/v70/guo17a.html |
| Angelopoulos, A.N. and Bates, S. (2021) A Gentle Introduction to Conformal Prediction. arXiv. | https://arxiv.org/abs/2107.07511 |
| Romano, Y., Patterson, E. and Candès, E.J. (2019) Conformalized Quantile Regression. NeurIPS. | https://arxiv.org/abs/1905.03222 |
| Cormack, G.V., Clarke, C.L.A. and Büttcher, S. (2009) Reciprocal Rank Fusion. SIGIR. | https://doi.org/10.1145/1571941.1572114 (BOT-BLOCK; META ok) |
| Robertson, S. and Zaragoza, H. (2009) The Probabilistic Relevance Framework: BM25 and Beyond. FnTIR. | https://doi.org/10.1561/1500000019 (BOT-BLOCK; META ok) |

---

## C. Competing / related tools (HTTP checked; facts only as stated on their pages)
| Tool | URL | Verified facts (24 Nov 2026) | Gap FundMatch UK targets |
|---|---|---|---|
| UKRI Gateway to Research | https://gtr.ukri.org/ | Free public search + API of funded UKRI projects (158,710) | Keyword search of past awards; does not read an abstract, predict a council or give award ranges |
| UKRI Funding Finder | https://www.ukri.org/opportunity/ | Free list of open/upcoming UKRI calls with council/status filters | Lists calls; no abstract matching or similar-funded evidence |
| NIH RePORTER Matchmaker | https://reporter.nih.gov/matchmaker | Free; text-in matching (cited in report) | US/NIH only |
| Dimensions (Digital Science) | https://www.dimensions.ai/ | Free version "for personal, non-commercial use"; paid platform links publications with grants, patents, trials | Full grant analytics behind subscription; not an applicant council recommender |
| Pivot-RP (Clarivate) | https://clarivate.com/academia-government/scientific-and-academic-research/research-funding-analytics/pivot-rp-funding/ · https://pivot.proquest.com/about_pivot | Institutional subscription; funding opportunity database + editorial researcher profiles and alerts (search result; Clarivate page bot-blocked) | Paywalled; profile-based (personal data); opportunity-focused not past-award evidence |
| Research Professional (Clarivate) | https://www.researchprofessional.com/ | Sign-in via university or subscriber account | Paywalled; news + calls, no abstract classifier |
| GrantForward | https://www.grantforward.com/ | "Search engine for federal grants, foundation grants…"; free trial / institution trial; "over 30,000" sponsors' opportunities | US-centric, subscription |
| Instrumentl | https://www.instrumentl.com/pricing | Paid plans with 14-day free trial; 36K+ active grants, 550k+ funder profiles; aimed at nonprofits/grant professionals | US foundation focus, not UK research councils |
| OpenAlex | https://openalex.org/ | Open scholarly graph incl. funders; free API key required since 13 Feb 2026 | Bibliographic, not funder recommendation |
| Semantic Scholar | https://www.semanticscholar.org/ · API https://www.semanticscholar.org/product/api | Free literature search + API | Papers, not grants |
| EU Funding & Tenders / CORDIS | https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/home · https://cordis.europa.eu/ | EU calls and funded-project records | EU programmes; no UK council mapping |
| GtR-Extended (research dataset) | https://github.com/wrmthorne/GtR-Extended | Unified UK funding-lifecycle DB (2026) | Dataset for researchers, not an applicant-facing tool; cannot be training data (FR-5) |

Market caveat: this is a desk check of public pages, not an exhaustive market survey; "we did not find" is the strongest honest claim.

---

## D. Potential academic contacts / related research (NO confirmed collaborations)
| Region | Institution · unit | Person (current role as verified) | Verified page | Example publication | Relevance |
|---|---|---|---|---|---|
| UK | University of Aberdeen · Dept of Computing Science (AberdeenML) | Dr Mingjun Zhong — Lecturer in ML; teaches CS5079; Director MSc IT | https://www.abdn.ac.uk/people/mingjun.zhong | (course coordinator) | Course coordinator — formative feedback only via normal channels |
| UK | University of Aberdeen · Computing Science | Dr Dewei Yi — Senior Lecturer; Director MSc AI | https://www.abdn.ac.uk/people/dewei.yi | (course lecturer) | Embedded/edge AI unit; federated learning supervision interests |
| UK | UCL STEaPP / Research on Research Institute (RoRI) | Prof James Wilsdon — Professor of Research Policy; Executive Director RoRI | https://profiles.ucl.ac.uk/93009-james-wilsdon · https://researchonresearch.org/ | The Metric Tide (2015) | Responsible metrics; funder experimentation |
| UK | University of Sheffield · Computer Science, NLP group (GATE) | Dr Diana Maynard — Senior Research Fellow, NLP group | https://sheffield.ac.uk/cs/people/research-staff/diana-maynard | GtR-Extended (2026) | Closest UK work on GtR + funding lifecycle |
| UK | University of Sheffield · Information, Journalism and Communication | Prof Mike Thelwall — Professor of Data Science | https://sheffield.ac.uk/ijc/people/academic-staff/mike-thelwall | Kousha & Thelwall (2023) | AI in research assessment, scientometrics |
| UK | University of Sussex · SPRU | Dr Ohid Yaqub — Associate Professor (per Sussex profile/search) | https://profiles.sussex.ac.uk/p141998-ohid-yaqub | Yaqub, Coburn & Moore (2024) Research Policy | Research funding styles/targeting |
| UK | The Alan Turing Institute | (institution only — no individual verified) | https://www.turing.ac.uk/ (BOT-BLOCK) | — | National AI institute; possible seminar audience |
| Europe | Leiden University · CWTS | Prof Ludo Waltman — Professor of Quantitative Science Studies; Scientific Director CWTS (from 1 Jan 2024) | https://www.universiteitleiden.nl/en/staffmembers/ludo-waltman | Leiden Manifesto (2015); VOSviewer (2010) | Science mapping, responsible metrics |
| Europe | Leiden University · CWTS | Nees Jan van Eck (VOSviewer co-author; current title not re-verified) | https://www.cwts.nl/ | VOSviewer (2010) | Topic maps for trend view |
| Europe | Fraunhofer ISI · Innovation and Knowledge Economy | Dr Rainer Frietsch — head of department (per ISI pages/search) | https://www.isi.fraunhofer.de/en/competence-center/innovations-wissensoekonomie.html | Neuhäusler & Frietsch (2012) | Bibliometric/patent indicators |
| Europe | TU Darmstadt · UKP Lab | Prof Iryna Gurevych — Full Professor, founder/head UKP | https://www.informatik.tu-darmstadt.de/ukp/ukp_home/head_ukp/index.en.jsp | Sentence-BERT (2019) | Embedding methods used in FundMatch |
| US | Allen Institute for AI (Ai2) · Semantic Scholar | Doug Downey, Sergey Feldman (listed in Ai2 team directory); Iz Beltagy (listed; personal site also names spiffy.ai) | https://allenai.org/team | SPECTER (2020), SciRepEval/SPECTER2 (2023), SciBERT (2019) | Scientific document embeddings |
| US | Yale · Computer Science (Yale NLP) | Arman Cohan — Assistant Professor | https://engineering.yale.edu/research-and-faculty/faculty-directory/arman-cohan | SPECTER; SciRepEval | Scientific NLP |
| US | (former Ai2) | Kyle Lo — personal site states "formerly a research scientist at Ai2… now working on something new" | https://kyleclo.com/ | SciBERT; S2ORC | Do not list as Ai2 |
| US | Northwestern · Kellogg CSSI | Prof Dashun Wang — founding director CSSI | https://www.kellogg.northwestern.edu/academics-research/faculty/wang_dashun/ | The Science of Science (2021) | Science of science, funding outcomes |
| US | Indiana University · Luddy School, CNS Center | Prof Katy Börner — Victor H. Yngve Distinguished Professor; founding director CNS | https://luddy.iu.edu/people/b%C3%B6rner-katy.html · https://cns.iu.edu/ | Fortunato et al. (2018) | Science maps, funding visualisation |
| Africa | Stellenbosch University · CREST / SciSTIP (DSI-NRF CoE in Scientometrics & STI Policy) | Prof Johann Mouton — CREST professor; director SciSTIP | https://www0.sun.ac.za/crest/staff/mouton/ (timeout to bot; confirmed by search) | Mouton (2019) Science-granting councils in Sub-Saharan Africa | African research funding systems |
| Africa | University of Pretoria · Computer Science, Data Science for Social Impact | Prof Vukosi Marivate — ABSA UP Chair of Data Science | https://www.cs.up.ac.za/research/dsfsi/ · https://www.up.ac.za/computer-science/staff-profiles/vukosi-marivate (BOT-BLOCK) | Nekoto et al. (2020) Masakhane | Low-resource NLP; transfer beyond UK |
| Africa | Masakhane (grassroots NLP community) | community | https://www.masakhane.io/ | Nekoto et al. (2020) | Multilingual extension |
| Africa | African Institute for Mathematical Sciences (AIMS) | institution only | https://aims.ac.za/ | — | Possible user testers / ML community |

Omitted because not verified today: University of Washington NLP individuals, ETH Zurich/EPFL individuals, Turing Institute named staff.

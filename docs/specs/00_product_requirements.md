# Product requirements: UKRI funding matcher (Idea A)

These requirements are for Idea A, the one I recommended. The choice of idea is still open; if you pick Idea B, I'll rewrite this tab.

The IDs (FR, NFR, F) are there so the report and notebook can refer back to each requirement. The **Brief** column shows which part of the CS5079 Task 1 marking each requirement supports. Numeric targets such as accuracy and response time are design targets I've proposed. They are not measured yet, so adjust them once you have results.

## 1. Product summary

- **Working name:** FundMatch UK (or another title from Ideas.docx).
- **Users:** early-career researchers, PhD applicants, university research offices, and small companies applying to Innovate UK.
- **Problem:** applicants don't know which UKRI council fits their idea, how big a typical award is, or what similar work has already been funded. The market check found no tool advertising these three things together (see the Market and gaps tab).
- **What it does:** the user pastes a title and abstract. The app then returns:
  - the most likely funder, with probabilities
  - a typical funding range
  - the 10 most similar funded projects, with links to Gateway to Research (GtR)
- **Data:** UKRI GtR projects, collected with your own code through the public GtR API (no key needed; 158,710 projects in our test call on 20 Nov 2026).

## 2. Functional requirements

### 2.1 Data collection

| ID | Requirement | How you check it | Priority | Brief |
| --- | --- | --- | --- | --- |
| FR-1 | Collect projects with your own Python code from the GtR API (`https://gtr.ukri.org/gtr/api/projects`, header `Accept: application/vnd.rcuk.gtr.json-v7`), paging through all results | Notebook cell runs end to end and saves raw JSON | Must | Data collection, 3 marks for own code |
| FR-2 | For each project, store: ID, title, abstract, lead funder, grant category, start and end dates, award amount, lead organisation, and GtR URL | Schema printed in notebook; missing-value counts reported | Must | Data collection |
| FR-3 | Respect the API: about 1 to 2 requests a second (limit is 10), retry with back-off on errors, and resume from the last page saved | Code shows rate limit, retry and checkpoint | Must | Data collection; reproducibility |
| FR-4 | Save a dated snapshot (for example Parquet) and record when it was collected and how many projects | Collection log in notebook | Must | Reproducibility |
| FR-5 | Do not use Kaggle or any ready-made GtR dataset (such as GtR-Extended) as training data. Mention them only for comparison | Stated in report | Must | Brief: "not allowed to use any data that is already prepared" |

### 2.2 Preprocessing and features

| ID | Requirement | How you check it | Priority | Brief |
| --- | --- | --- | --- | --- |
| FR-6 | Remove duplicates, empty or very short abstracts, and placeholder text; report how many rows each step removes | Before-and-after counts table | Must | Preprocessing, 3 marks |
| FR-7 | Normalise funder labels (EPSRC, ESRC, MRC, BBSRC, AHRC, NERC, STFC, Innovate UK, others); merge or drop very rare classes and say why | Class distribution plot | Must | Preprocessing |
| FR-8 | Filter by date so funding amounts are comparable, and exclude projects too recent to have an outcome | Filter rule stated with counts | Must | Preprocessing |
| FR-9 | Build features: TF-IDF for baselines; SciBERT, MiniLM or SPECTER embeddings for transformer models; log of award amount plus funder, category, year and duration for the funding model | Feature shapes printed | Must | Feature engineering |
| FR-10 | Split train, validation and test with stratification by funder, and keep a time-based test set as an extra check | Split sizes and class balance shown | Must | Training process |
| FR-11 | Explain why the data is large enough: number of projects, examples per class, and a learning curve | Learning curve plot | Must | Project definition, 2 marks for data sufficiency |

### 2.3 Models and training

| ID | Requirement | How you check it | Priority | Brief |
| --- | --- | --- | --- | --- |
| FR-12 | Funder classifier: compare TF-IDF with Logistic Regression, Linear SVM and Complement NB against fine-tuned SciBERT | Results table: macro-F1, top-3 accuracy, confusion matrix | Must | Candidate models, 3 marks |
| FR-13 | Similar projects: compare BM25 or TF-IDF against MiniLM and SPECTER embeddings, searched with FAISS | Recall@10 and MRR on same-topic labels, plus about 50 queries judged by hand | Must | Candidate models |
| FR-14 | Funding range: compare Ridge against LightGBM on log amount, and give a range (for example 25th to 75th percentile of similar projects, or quantile regression) | MAE on log amount; range coverage on test set | Should | Candidate models |
| FR-15 | Handle imbalance with class weights or resampling, and report per-class results | Per-class F1 table | Must | Training process |
| FR-16 | Tune hyperparameters on validation data only; record seeds, settings and training time | Settings table in notebook | Must | Training process, 3 marks |
| FR-17 | Choose the deployed model by weighing accuracy, speed, model size and memory on the free host; justify the choice against the runners-up | Comparison table plus written justification | Must | Deployment choice, 6 marks |
| FR-18 | Error analysis: inspect misclassified examples (for example EPSRC versus Innovate UK) and explain them | Short write-up with examples | Should | In-depth analysis |

### 2.4 The app (user-facing functions)

| ID | Requirement | How you check it | Priority | Brief |
| --- | --- | --- | --- | --- |
| FR-19 | Text inputs for title and abstract, with a minimum length check and a clear message if the text is too short | Manual test with empty and short input | Must | Deployment |
| FR-20 | Show the top funders as a probability bar chart, with a warning when the top probability is low | Screenshot in report | Must | Deployment |
| FR-21 | Show the 10 most similar funded projects: title, funder, amount, dates, lead organisation and a GtR link | Links open the correct GtR pages | Must | Deployment |
| FR-22 | Show a typical funding range for similar projects, with the caveat that it is not a quote | Screenshot | Must | Deployment |
| FR-23 | Show a funding trend chart for the topic by year and council | Screenshot | Should | Deployment, "excellent functionalities" |
| FR-24 | Provide example abstracts the user can load with one click, for demos and markers | Button works on the deployed app | Should | Showcasing deployment |
| FR-25 | Let users filter similar projects by council or year range | Filter changes the results | Could | Deployment |
| FR-26 | Let users download results as CSV | File downloads | Could | Deployment |
| FR-27 | Show an "About" panel: data source and date, model used, test accuracy, limits, and the not-official-advice notice | Visible on the app | Must | Deployment; ethics |

## 3. Non-functional requirements

| ID | Category | Requirement | Target or check | Priority |
| --- | --- | --- | --- | --- |
| NFR-1 | Accuracy | The deployed funder model clearly beats the majority-class and TF-IDF baselines | Proposed target: macro-F1 above the best baseline, and top-3 accuracy reported. Set the final figure from results | Must |
| NFR-2 | Speed | Results appear quickly on the free host | Proposed target: under 3 seconds per query after the app has loaded. Measure and report | Must |
| NFR-3 | Resource use | Model, embeddings and FAISS index fit within the free hosting tier's memory and storage limits | Check the current Streamlit Community Cloud limits before choosing the model; quantise or shrink embeddings if needed | Must |
| NFR-4 | Availability | The public link works when the marker opens it | Test the link on submission day; document how to wake a sleeping app | Must |
| NFR-5 | Reproducibility | Another student can rerun collection and training from the notebook and README | Pinned `requirements.txt`, fixed seeds, saved snapshot; the notebook is submitted with outputs kept | Must |
| NFR-6 | Code quality | Code is documented, commented and organised into clear functions | Docstrings, comments, no unused cells | Must |
| NFR-7 | Usability | A first-time user understands results without help | Plain-English labels, tooltips on charts, example inputs | Should |
| NFR-8 | Accessibility | Readable charts and text | Colour-blind-safe palette, labelled axes, enough contrast | Should |
| NFR-9 | Privacy | No personal data is ranked or profiled | Show organisations and GtR links only; never researcher names; do not store user abstracts | Must |
| NFR-10 | Licensing and attribution | Follow the Open Government Licence and credit all sources and borrowed code | OGL attribution on the app; no suggestion of UKRI endorsement; sources cited in code and report | Must |
| NFR-11 | Transparency | Users can see how good the model is and where it fails | Test metrics and known weak classes in the About panel | Should |
| NFR-12 | Maintainability | Data can be refreshed later without rewriting code | Collection script can be rerun; models reload from saved files | Could |
| NFR-13 | Security | No secrets in the repository | The GtR API needs no key; any hosting tokens go in Streamlit secrets | Must |
| NFR-14 | Report limits | The report fits 1,500 words and maps each answer to a brief question | Word count checked; headings follow the brief | Must |

## 4. Feature list

**Must have (minimum for a strong grade)**

- **Funder predictor:** title and abstract in, ranked UKRI councils with probabilities out.
- **Similar funded projects:** the top 10, with GtR links, amounts and dates.
- **Typical funding range:** based on similar past awards.
- **About and limits panel:** data date, model, accuracy, and the "not official UKRI advice" notice.
- **Model comparison in the notebook:** baselines against transformers for each task.

**Should have (to reach the top band's "excellent functionalities")**

- Topic funding trend chart by year and council.
- One-click example abstracts.
- Low-confidence warning and per-class accuracy shown to users.
- Error analysis section.

**Could have (if time allows)**

- Filters by council and year.
- CSV download of results.
- A link from each predicted council to its live calls on the UKRI Funding finder. This answers the main market gap that the tool looks at past awards, not open calls.

**Won't have (out of scope, so say so in the report)**

- Predicting whether a grant will be funded.
- Writing or editing proposals.
- Ranking or profiling individual researchers.

## 5. Map to the marking scheme

| Brief section | Marks | Requirements that cover it |
| --- | --- | --- |
| Project definition: usefulness | 2 | Product summary, market gaps |
| Project definition: uses data | 1 | FR-1, FR-2 |
| Project definition: why the data is sufficient | 2 | FR-11 |
| Data collection: own sources | 2 | FR-1, FR-5 |
| Data collection: own code | 3 | FR-1, FR-3, FR-4 |
| Preprocessing and feature engineering | 3 | FR-6 to FR-9 |
| Candidate models and why | 3 | FR-12 to FR-14 |
| Training process | 3 | FR-10, FR-15, FR-16 |
| Why this model is deployed | 6 | FR-17, NFR-1 to NFR-3 |
| Deployment, showcase and link | 5 | FR-19 to FR-27, NFR-4 |
| Report quality, reproducibility, code quality | (overall criteria) | NFR-5, NFR-6, NFR-14 |

## Notes

- The GtR record count and API details come from our test call on 20 Nov 2026. Recheck the count when you collect.
- Whether every GtR project carries an award amount, and how the funding records link to projects, is not confirmed yet. Check this early in collection, because FR-14 and FR-22 depend on it.
- Hosting limits change, so check them before fixing the model choice (NFR-3).

# DataMetaMap Project Plan

## Project Goal
DataMetaMap aims to compare datasets within a unified vector space to identify semantic similarities. The core idea is that if a model performs well on one dataset, it will likely perform well on semantically similar datasets nearby in embedding space.

---

## Development Phases & Tasks

### Phase 1: Research and Preparation
- **Literature Review**  
  Study existing methods for dataset embedding, similarity measurement, and transferability estimation to identify best practices.

- **Data Collection**  
  Gather a diverse collection of datasets for experimentation, ensuring they represent various domains and formats.

- **Planning and Specifications**  
  Define technical specifications and success criteria based on research findings and data availability.

---

### Phase 2: Implementation and Testing
- **Core Algorithm Development**  
  Implement algorithms to embed datasets into a shared vector space and compute similarity metrics between them.

- **Testing and Quality Assurance**  
  Develop unit and integration tests to validate correctness, reliability, and performance of the implemented methods.

- **Benchmarking and Visualization**  
  Run benchmarks on collected datasets and produce visual outputs such as similarity matrices to analyze and interpret results.

---

### Phase 3: Documentation and Dissemination
- **Technical Report**  
  Document the methodology, experimental setup, and findings in a comprehensive technical report.

- **User and Developer Documentation**  
  Create detailed documentation for users and contributors, including setup guides and API references.

- **Demo Examples and Blog Post**  
  Prepare example notebooks or scripts demonstrating real-world use cases, and write an explanatory blog post highlighting project value and insights.

## Remastered 

### Phase 1: Research and Preparation
- **Literature Review**  
  Study existing methods for dataset embedding, similarity measurement, and transferability estimation to identify best practices.

- **Baseline Selection**  
  Identify and select baseline methods from literature for comparison during benchmarking.
  
  Description (done by Meshkov Vladislav):
    - Establish baselines for each embedding method as specified in the paper
    - Assess baselines from the literature and determine their appropriateness for our benchmarking framework
    - Conduct a literature review to identify similar papers and gather additional straightforward baselines for meaningful comparison
    - Document baseline descriptions in the benchmark specifications, along with rationale for their inclusion

- **Data Collection**  
  Gather a diverse collection of datasets for experimentation, ensuring they represent various domains and formats.

- **Data Preprocessing Pipeline**  
  Design and implement preprocessing steps to handle different dataset formats and ensure consistent input for embedding methods.


  Description (done by Minashkin Vladislav):
    - Handle diverse data types: images, text, tabular, and time series with type-specific loaders
    - Fill missing values and remove bad data points
    - Save all settings for exact reproduction

- **Evaluation Metrics Definition**  
  Define quantitative metrics to evaluate embedding quality and similarity measurement accuracy.
  
  Description (done by Stepanov Ilya):
    - Define cosine similarity, Euclidean distance and kernel-based distance as core metrics to evaluate geometric separability and structural relationships between dataset embeddings in the latent space
    - Define Maximum Mean Discrepancy (MMD) metric as described in the original paper
    - Ensure that all embedding methods and baselines will be evaluate using all metrics so that comparison across methods is consistent and reproducible


- **Planning and Specifications**  
  Define technical specifications and success criteria based on research findings and data availability.

---

### Phase 2: Implementation and Testing
- **Core Algorithm Development**  
  Implement algorithms to embed datasets into a shared vector space and compute similarity metrics between them.

- **Baseline Implementations**  
  Implement selected baseline methods from literature for comparison.

- **Testing and Quality Assurance**  
  Develop unit and integration tests to validate correctness, reliability, and performance of the implemented methods.

- **Performance Optimization**  
  Profile and optimize code for memory efficiency and computational speed, especially for large datasets.

- **Error Handling and Logging**  
  Implement robust error handling and logging mechanisms for debugging and monitoring.

- **Benchmarking and Visualization**  
  Run benchmarks on collected datasets and produce visual outputs such as similarity matrices to analyze and interpret results.

---

### Phase 3: Documentation and Dissemination
- **Technical Report**  
  Document the methodology, experimental setup, and findings in a comprehensive technical report.

- **User and Developer Documentation**
  Description (done by Papay Ivan)
      - create detailed documentation for users and contributors, including setup guides and API references
      - create github.io page where user can find documentation for all classes and their methods
      - Github.io page must have headers for functions and links to their each source code. 

- **Demo Examples and Blog Post**  
  Prepare example notebooks or scripts demonstrating real-world use cases, and write an explanatory blog post highlighting project value and insights.

- **Benchmark Results Repository**  
  Publish benchmark results, precomputed embeddings, and similarity matrices in a public repository for reproducibility.

- **Future Work Roadmap**  
  Outline potential extensions, improvements, and research directions based on current findings.

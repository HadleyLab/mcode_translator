# MCODE Translator System Architecture

## Current Architecture

```mermaid
graph TD
    A[User Interface] --> B[CLI Interface]
    A --> C[Web UI]
    B --> D[Optimization Framework]
    C --> D
    D --> E[Prompt Library]
    D --> F[Model Library]
    D --> G[Test Cases]
    D --> H[Gold Standard Data]
    E --> I[Prompt Loader]
    F --> J[Model Loader]
    D --> K[LLM Providers]
    D --> L[Benchmark Results]
```

## Enhanced Architecture with Default Prompt Functionality

```mermaid
graph TD
    A[User Interface] --> B[New Unified CLI]
    A --> C[Enhanced Web UI]
    B --> D[Optimization Framework]
    C --> D
    D --> E[Enhanced Prompt Library]
    D --> F[Model Library]
    D --> G[Test Cases]
    D --> H[Gold Standard Data]
    E --> I[Enhanced Prompt Loader]
    F --> J[Model Loader]
    D --> K[LLM Providers]
    D --> L[Benchmark Results]
    D --> M[Default Prompt Manager]
    M --> E
```

## CLI Command Structure

```mermaid
graph TD
    A[mcode-optimize] --> B[run]
    A --> C[set-default]
    A --> D[view-results]
    A --> E[list-prompts]
    A --> F[show-prompt]
    A --> G[benchmark]
    A --> H[validate]
    
    B --> B1[test-cases]
    B --> B2[gold-standard]
    B --> B3[output]
    B --> B4[metric]
    B --> B5[top-n]
    
    C --> C1[prompt-type]
    C --> C2[prompt-name]
    C --> C3[from-results]
    C --> C4[metric]
    C --> C5[results-dir]
    
    D --> D1[results-dir]
    D --> D2[metric]
    D --> D3[top-n]
    D --> D4[format]
    D --> D5[export]
    
    E --> E1[type]
    E --> E2[status]
    E --> E3[default-only]
    E --> E4[config]
```

## Data Flow for Optimization Process

```mermaid
sequenceDiagram
    participant U as User
    participant C as CLI
    participant F as Optimization Framework
    participant P as Prompt Library
    participant M as Model Library
    participant T as Test Cases
    participant G as Gold Standard
    participant L as LLM Provider
    participant R as Results Storage

    U->>C: mcode-optimize run
    C->>F: Initialize optimization
    F->>P: Load prompts
    F->>M: Load models
    F->>T: Load test cases
    F->>G: Load gold standard
    loop For each prompt-model-test combination
        F->>L: Execute benchmark
        L-->>F: Return results
        F->>R: Save benchmark result
    end
    F->>F: Calculate metrics
    F->>F: Determine best combinations
    C->>C: Display results
    U->>C: mcode-optimize set-default
    C->>F: Get best combination
    F->>P: Update default prompt
    P->>P: Save configuration
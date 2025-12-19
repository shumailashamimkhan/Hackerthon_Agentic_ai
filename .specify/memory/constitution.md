<!--
SYNC IMPACT REPORT
Version change: N/A -> 1.0.0 (Initial creation)
Added sections: All principles and sections as specified for the AI-Native Textbook project
Removed sections: None (this is an initial creation)
Templates requiring updates:
- ✅ plan-template.md - Will need to ensure the Constitution Check aligns with new principles
- ✅ spec-template.md - Aligned with new requirements
- ✅ tasks-template.md - Aligned with new task categorization reflecting new principles
Runtime guidance docs: README.md not found - no updates needed
Follow-up TODOs: None - all placeholders have been filled
-->
# AI-Native Textbook for Physical AI & Humanoid Robotics Constitution

## Core Principles

### I. Interactive Education Focus
Every feature must enhance the educational experience of the Physical AI & Humanoid Robotics course. All implementations should prioritize user engagement and learning outcomes over technical complexity. Products must feel like a real AI-powered education platform.

### II. Clean Architecture & Modularity
Maintain simple, readable code with clean separation of concerns. Backend must be modular (FastAPI + services + routes), frontend extremely simple. Use clean folder structure: /backend, /website, /rag, /agents.

### III. Performance & Accessibility
Products must load quickly, work on mobile devices, support low-end devices, and feel responsive. All features must work on free tiers (Qdrant + Neon). No unnecessary animations or dependencies that could slow performance.

### IV. Functional Completeness
Each core deliverable must be fully functional: Docusaurus-based textbook with 6-8 chapters, RAG chatbot answering questions ONLY from the book, user authentication (Better-Auth), personalized content, Urdu translation, auto-generated summaries/quizzes.

### V. Grounded AI Interactions
AI features (RAG chatbot, personalization, content generation) must be accurate, cited, and grounded in the source material. Implement proper chunking and embeddings to ensure high accuracy.

### VI. Deployability & Monitoring
Systems must be deployable within 90 seconds with URLs for frontend (Vercel), backend (Railway), vectors (Qdrant), and database (Neon). Include health checks and logging to monitor backend errors.

## Technology Stack Requirements
Backend: FastAPI, Services architecture
Frontend: Docusaurus
Authentication: Better-Auth
Vector DB: Qdrant
Database: Neon
Deployment: Frontend → Vercel, Backend → Railway
Translation: One-click Urdu translation
AI Features: RAG, Personalization, Auto-generation

## Development Workflow
TDD approach for critical functionality
Code reviews focusing on compliance with principles
Performance testing for mobile/low-end devices
Testing of all AI features for accuracy and grounding

## Governance
All PRs/reviews must verify compliance with the six core principles.
Amendments require team discussion and consensus.
Complexity must be justified with clear learning or technical benefits.

**Version**: 1.0.0 | **Ratified**: 2025-12-18 | **Last Amended**: 2025-12-18

\# 📊 DataPilot - AI Data Analysis Agent



DataPilot is a general-purpose AI data analysis agent that allows users to upload a CSV dataset and ask questions about the data using natural language.



Instead of asking an AI model to guess or calculate numerical answers directly, DataPilot uses Python and Pandas to perform deterministic calculations on the uploaded dataset. Gemini is used to understand the user's question, select the appropriate analysis operation, and explain the verified result.



\---



\## 🚀 Features



\- Upload arbitrary CSV datasets

\- Automatically inspect dataset structure and schema

\- Detect numeric, categorical, datetime, and identifier columns

\- Ask questions using natural language

\- Perform deterministic calculations using Python/Pandas

\- Support:

&#x20; - totals

&#x20; - averages

&#x20; - minimums

&#x20; - maximums

&#x20; - grouping

&#x20; - filtering

&#x20; - ranking

&#x20; - filtered aggregation

&#x20; - quarterly analysis

\- Display the computed evidence/table behind the answer

\- Maintain conversation context during the session

\- Streamlit interface for easy interaction

\- Automated tests for the analysis engine and schema detection



\---



\## 🧠 How DataPilot Works



DataPilot separates natural-language understanding from numerical computation.



```text

&#x20;                   User

&#x20;                    |

&#x20;                    v

&#x20;             Natural-language

&#x20;                 question

&#x20;                    |

&#x20;                    v

&#x20;                Gemini

&#x20;                    |

&#x20;             Understand intent

&#x20;                    |

&#x20;                    v

&#x20;            Select analysis tool

&#x20;                    |

&#x20;                    v

&#x20;             Python / Pandas

&#x20;                    |

&#x20;             Deterministic

&#x20;               calculation

&#x20;                    |

&#x20;                    v

&#x20;             Evidence / Result

&#x20;                    |

&#x20;                    v

&#x20;                Gemini

&#x20;                    |

&#x20;             Explain result

&#x20;                    |

&#x20;                    v

&#x20;              Final answer


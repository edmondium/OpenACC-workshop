# Task 4: Data Copies

To be super portable (and understand data movement behavior) we removed managed memory transfers (no Unified Memory). We need to manually add `copy` clauses to the OpenACC regions now. Do so, as indicated by the TODOs.

Compile the program with `make`, submit it to the batch system with `make run`.

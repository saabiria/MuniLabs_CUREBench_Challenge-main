## What is new or modified
1. I created a virtual environment (curebench-env) to install all the dependencies (requirements).
Install using;
```bash
pip install -r requirements.txt
```

2. I created a models directory (folder) containing different models for evaluation. These include;
>Deep Seek Model
>Gemini Model
>Llama Model
>Phi Model
*The current models implemented in the agent are the Deep Seek model to prevent limitations of tokenisation issues.
In this same directory, I added an agent (my_agent.py) that calls the main agent(s) in the agents directory.

3. I created a models registration file (config.py) to register all the models used by the agent. 

4. I created the dataset directory to store the CUREbench validation and test datasets downloaded from "https://www.kaggle.com/competitions/cure-bench/data' from the data explorer section of the page.

5. I created the agents directory to store agents. The main agent (clinical_agent.py) helps the model to decide. The (planner_agent.py) maps question types to steps.
6. I created a retrieval directory to store the knowledge retriever (medical_retriever.py). It helps in searching relevant medical knowledge. 

7. I created a tools directory to store the medical tools (drug_lookup.py, interaction_checker.py, tool_manager.py). I use these tools instead of the ToolKit, which is difficult to install and configure.

8. I created the results directory to store the output of my benchmark evaluation. 

9. I modified the metadata_config_val.json updating the following fields ( model_name, model_type, track, base_model_name, dataset and dataset_path) and I added an output path.  I modified metadata_config_test.json in the following fields (model_name, model_type, track, base_model_name, dataset, additional_info, and dataset_path), and I added to the output file as well.

10. I created an evaluation directory to run the evaluations.

11. I modified CustomModel.inference in the eval_framework.py

12. I modified the model name in the run.py
13. 11/April/2026 I updated agents/clinical_agent.py, planner_agent.py,    retrieval/medical_retriever.py for improved accuracy (75%-80%)


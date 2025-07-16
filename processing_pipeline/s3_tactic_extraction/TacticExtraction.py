import pandas as pd

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from processing_pipeline.s3_tactic_extraction.tactics.tactic_description_full import tactic_descriptions_full
from processing_pipeline.s3_tactic_extraction.tactics.tactic_list_simplified import TacticSimplifiedModelResponse
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.BaseStage import BaseStage


class TacticExtractionStage(BaseStage):
    data_model = TacticSimplifiedModelResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    cache_dir = AbsDirPath.CACHE / FolderNames.TACTIC_EXTRACTION_DIR
    in_dir = AbsDirPath.S2_ARCH_RELEVANCE_CHECK
    out_dir = AbsDirPath.S3_TACTIC_EXTRACTION
    stage_name = 's3'

    tactic_descriptions_list_simplified = "\n".join(
        f"- {tactic}: {details["description"]}"
        for tactic, details in tactic_descriptions_full.items())

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
You are an expert in software architecture tactics. Your task is to analyze the given text and categorize it according to software architecture tactics, quality attributes, and system responses.

### **Key Concepts**
- **Tactics**: Design decisions that influence system responses to achieve quality attributes.
- **Response**: The system's reaction to a stimulus, either at runtime or development time.

### **Available Quality Attributes**
- **Availability**: Minimizing service outages by masking or repairing faults.
- **Interoperability**: Enabling seamless data exchange between systems.
- **Modifiability**: Facilitating easy changes for new features, bug fixes, or adaptations.
- **Performance**: Ensuring the system meets timing and throughput requirements.
- **Security**: Protecting data from unauthorized access and ensuring confidentiality.
- **Testability**: Making software easy to test and debug.
- **Usability**: Enhancing user experience and reducing operational errors.
- **Energy Efficiency**: Reducing energy consumption in software and hardware.

### **Your Task**
For the given text:
1. Determine the **specific tactic** from the list below and identify the **quality attribute** it addresses
2. Assign the tactic to `tactic` field. 
3. Describe the **system's response** to the stimulus, assign it to `response` field

### **Examples**
- **Availability**  
  - **Stimulus**: Server becomes unresponsive.  
  - **Tactic**: Heartbeat Monitor (Detect Faults).  
  - **Response**: Inform Operator, Continue to Operate.  
  - **Response Measure**: No Downtime.  

- **Performance**  
  - **Stimulus**: Users initiate transactions.  
  - **Tactic**: Increase Resources (Manage Resources).  
  - **Response**: Transactions Are Processed.  
  - **Response Measure**: Average Latency of Two Seconds.  

- **Security**  
  - **Stimulus**: Disgruntled employee attempts to modify the pay rate table.  
  - **Tactic**: Maintain Audit Trail (React to Attacks).  
  - **Response**: Record attempted modification.  
  - **Response Measure**: Time taken to restore data.  

### **Available Tactics**
{cls.tactic_descriptions_list_simplified}

### **Analyze the Following Text**
"{x['sentence']}"
"""


def main():
    TacticExtractionStage(hostname=LLMHost.GREEN_LAB).execute(["root-project"], reverse=True)


if __name__ == "__main__":
    main()

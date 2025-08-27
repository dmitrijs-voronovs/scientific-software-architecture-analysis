```python
from pathlib import Path

from constants.abs_paths import AbsDirPath
from processing_pipeline.model.CSVDFHandler import CSVDFHandler
from processing_pipeline.utilities.data_transformation import load_all_files, load_all_csv_files
```

# S0


```python
df = CSVDFHandler().read_df(AbsDirPath.SAMPLES_VERIFIED / "s0_manual_ver.csv")
# df = load_all_csv_files(AbsDirPath.SAMPLES_VERIFIED, name_contains="s0.part")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>sentence</th>
      <th>s0_prompt</th>
      <th>s0_to_eliminate</th>
      <th>s0_reasoning</th>
      <th>dmitry_approves</th>
      <th>s0_v_prompt</th>
      <th>s0_v_ground_truth_category</th>
      <th>s0_v_evaluation</th>
      <th>s0_v_reasoning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2041</td>
      <td>LoweringError: Failed in nopython mode pipelin...</td>
      <td>\nYou are an expert in analyzing and categoriz...</td>
      <td>True</td>
      <td>This is an error message from a program. It co...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>Log File / Trace / Terminal Output</td>
      <td>correct</td>
      <td>My verdict is correct because the ground truth...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12677</td>
      <td>Compute the static offset of the ultimate dest...</td>
      <td>\nYou are an expert in analyzing and categoriz...</td>
      <td>True</td>
      <td>This appears to be a technical description rel...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>Low-Level Implementation Note</td>
      <td>correct</td>
      <td>My verdict is correct because the ground truth...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2997</td>
      <td>[WIP] Parallelise reading of sub-branches in T...</td>
      <td>\nYou are an expert in analyzing and categoriz...</td>
      <td>False</td>
      <td>This content is a commit message that explains...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>Interactive Communication</td>
      <td>correct</td>
      <td>My verdict is correct because the ground truth...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13110</td>
      <td>Create a new node, and save it in the context ...</td>
      <td>\nYou are an expert in analyzing and categoriz...</td>
      <td>True</td>
      <td>This appears to be an instruction or comment a...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>Low-Level Implementation Note</td>
      <td>correct</td>
      <td>My verdict is correct because the ground truth...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1397</td>
      <td>e produces undefined behaviour because the sid...</td>
      <td>\nYou are an expert in analyzing and categoriz...</td>
      <td>False</td>
      <td>This content is primarily technical but contai...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>High-Level Instructional Guide</td>
      <td>correct</td>
      <td>My verdict is correct because the ground truth...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (384, 10)




```python
df.groupby("s0_v_evaluation").size().rename("count")
```




    s0_v_evaluation
    correct      256
    incorrect    128
    Name: count, dtype: int64




```python
df.groupby("s0_v_evaluation").size().rename("percentage") / df.shape[0] * 100
```




    s0_v_evaluation
    correct      66.666667
    incorrect    33.333333
    Name: percentage, dtype: float64




```python
df.dmitry_approves = df.dmitry_approves.astype(bool)
```


```python
df.groupby("dmitry_approves").size().rename("count")
```




    dmitry_approves
    False     74
    True     310
    Name: count, dtype: int64




```python
df.groupby("dmitry_approves").size().rename("percentage") / df.shape[0] * 100
```




    dmitry_approves
    False    19.270833
    True     80.729167
    Name: percentage, dtype: float64




```python
df.groupby(["dmitry_approves", "s0_v_evaluation"]).size().rename("count")
```




    dmitry_approves  s0_v_evaluation
    False            correct             12
                     incorrect           62
    True             correct            244
                     incorrect           66
    Name: count, dtype: int64



# S1


```python
df = CSVDFHandler().read_df(AbsDirPath.SAMPLES_VERIFIED / "s1_manual_ver.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>qa</th>
      <th>sentence</th>
      <th>s1_prompt</th>
      <th>s1_analysis_problem_vs_solution</th>
      <th>s1_analysis_mechanism_vs_feature</th>
      <th>s1_analysis_causal_link</th>
      <th>s1_analysis_rubric_check</th>
      <th>s1_true_positive</th>
      <th>s1_reasoning</th>
      <th>dmitry_approves</th>
      <th>s1_v_prompt</th>
      <th>s1_v_is_executor_reasoning_valid</th>
      <th>s1_v_evaluation</th>
      <th>s1_v_reasoning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>deployability</td>
      <td>Patch release of v6.26 series. [:spiral_notepa...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text is describing a release process and p...</td>
      <td>This describes a functional feature (a patch r...</td>
      <td>The causal link between the described action a...</td>
      <td>The content does not mention any specific mech...</td>
      <td>False</td>
      <td>The text does not describe an architectural me...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>True</td>
      <td>correct</td>
      <td>The executor's reasoning was valid because the...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36</td>
      <td>availability</td>
      <td>Shuffle script for training runs out of memory...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text describes a problem related to memory...</td>
      <td>Yes, it is describing a **problem** rather tha...</td>
      <td>The link between the problem and the quality a...</td>
      <td>This mechanism does not match the inclusion cr...</td>
      <td>False</td>
      <td>The analysis concludes that this text describe...</td>
      <td>0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>False</td>
      <td>incorrect</td>
      <td>The executor's reasoning was invalid because i...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>725</td>
      <td>deployability</td>
      <td>Receive message from a socket. This standalone...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text describes a problem related to how th...</td>
      <td>Yes, it is describing a specific implementatio...</td>
      <td>The link between the problem and the quality a...</td>
      <td>This mechanism (the specific implementation pa...</td>
      <td>True</td>
      <td>The text identifies a problem in how the funct...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>True</td>
      <td>correct</td>
      <td>The executor's reasoning was valid because the...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>328</td>
      <td>interoperability</td>
      <td>tegration region is divided into subintervals,...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text describes a mechanism for handling in...</td>
      <td>This is describing an architectural mechanism ...</td>
      <td>The text explicitly links the mechanism (`gsl_...</td>
      <td>The rubric for interoperability requires excha...</td>
      <td>True</td>
      <td>The text describes an architectural mechanism ...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>True</td>
      <td>correct</td>
      <td>The user's query is about a programming proble...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>deployability</td>
      <td>@sawenzel I get your point about thread safety...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text is primarily discussing issues relate...</td>
      <td>It mentions clang-formatting using ROOT style ...</td>
      <td>The causal link between these elements and dep...</td>
      <td>The rubric inclusion criteria require explicit...</td>
      <td>False</td>
      <td>The text does not provide concrete evidence of...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>True</td>
      <td>incorrect</td>
      <td>The text does not contain any specific mechani...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (384, 15)




```python
df.groupby("s1_v_evaluation").size().rename("percentage") / df.shape[0] * 100
```




    s1_v_evaluation
    correct      78.90625
    incorrect    21.09375
    Name: percentage, dtype: float64




```python
df.groupby("s1_v_evaluation").size().rename("count")
```




    s1_v_evaluation
    correct      303
    incorrect     81
    Name: count, dtype: int64




```python
df.groupby("s1_v_evaluation").get_group("incorrect")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>qa</th>
      <th>sentence</th>
      <th>s1_prompt</th>
      <th>s1_analysis_problem_vs_solution</th>
      <th>s1_analysis_mechanism_vs_feature</th>
      <th>s1_analysis_causal_link</th>
      <th>s1_analysis_rubric_check</th>
      <th>s1_true_positive</th>
      <th>s1_reasoning</th>
      <th>dmitry_approves</th>
      <th>s1_v_prompt</th>
      <th>s1_v_is_executor_reasoning_valid</th>
      <th>s1_v_evaluation</th>
      <th>s1_v_reasoning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>36</td>
      <td>availability</td>
      <td>Shuffle script for training runs out of memory...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text describes a problem related to memory...</td>
      <td>Yes, it is describing a **problem** rather tha...</td>
      <td>The link between the problem and the quality a...</td>
      <td>This mechanism does not match the inclusion cr...</td>
      <td>False</td>
      <td>The analysis concludes that this text describe...</td>
      <td>0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>False</td>
      <td>incorrect</td>
      <td>The executor's reasoning was invalid because i...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>deployability</td>
      <td>@sawenzel I get your point about thread safety...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text is primarily discussing issues relate...</td>
      <td>It mentions clang-formatting using ROOT style ...</td>
      <td>The causal link between these elements and dep...</td>
      <td>The rubric inclusion criteria require explicit...</td>
      <td>False</td>
      <td>The text does not provide concrete evidence of...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>True</td>
      <td>incorrect</td>
      <td>The text does not contain any specific mechani...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>336</td>
      <td>availability</td>
      <td>Hello @gunjanbaid , sorry for not replying soo...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text does not describe a problem or soluti...</td>
      <td>It is a feature of the BAM file processing mec...</td>
      <td>There is no direct causal link between the BAM...</td>
      <td>The described mechanism matches the inclusion ...</td>
      <td>True</td>
      <td>The text describes a BAM file processing mecha...</td>
      <td>0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>False</td>
      <td>incorrect</td>
      <td>The executor's reasoning was invalid because i...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>316</td>
      <td>deployability</td>
      <td>&gt; So, would the error still be there if you ch...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text discusses whether changing an example...</td>
      <td>This is a feature because it describes what ha...</td>
      <td>The link between the problem of ordering or de...</td>
      <td>This mechanism (static object initialization) ...</td>
      <td>True</td>
      <td>The analysis identifies that the problem lies ...</td>
      <td>0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>False</td>
      <td>incorrect</td>
      <td>The executor incorrectly classified an issue w...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>availability</td>
      <td>turned out that there was an error on my side.</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text describes a user's own mistake or mis...</td>
      <td>This statement is about an error, so it doesn'...</td>
      <td>There's no causal link to an availability mech...</td>
      <td>The content does not describe any inclusion cr...</td>
      <td>False</td>
      <td>This text snippet is describing a system probl...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>False</td>
      <td>incorrect</td>
      <td>The content does not contain any code or speci...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>360</th>
      <td>301</td>
      <td>usability</td>
      <td>Slot called whenever the stats fill style is m...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text does not present a bug or issue but d...</td>
      <td>It describes a specific feature (slot call) th...</td>
      <td>The link between the slot call and the stats f...</td>
      <td>This mechanism (slot calls) aligns with the us...</td>
      <td>True</td>
      <td>The text describes a specific feature (slot ca...</td>
      <td>0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>False</td>
      <td>incorrect</td>
      <td>The executor's reasoning was invalid because i...</td>
    </tr>
    <tr>
      <th>361</th>
      <td>731</td>
      <td>deployability</td>
      <td>Build only one PyROOT if PYTHON_EXECUTABLE or ...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text does not describe a solution but rath...</td>
      <td>No, it's not describing a mechanism. It's more...</td>
      <td>The link between the problem (missing configur...</td>
      <td>This mechanism matches the inclusion criteria ...</td>
      <td>False</td>
      <td>The analysis concludes that this is not a true...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>False</td>
      <td>incorrect</td>
      <td>The executor's reasoning was invalid because i...</td>
    </tr>
    <tr>
      <th>362</th>
      <td>965</td>
      <td>availability</td>
      <td>Reimplement TGraph::Sort using std::stable_sor...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text discusses reimplementing a function a...</td>
      <td>This is a feature because it describes the imp...</td>
      <td>The link between the changes and the problem i...</td>
      <td>This mechanism (reimplementing TGraph::Sort) m...</td>
      <td>True</td>
      <td>The text describes an architectural feature (i...</td>
      <td>0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>False</td>
      <td>incorrect</td>
      <td>The executor's reasoning was invalid because t...</td>
    </tr>
    <tr>
      <th>365</th>
      <td>351</td>
      <td>integrability</td>
      <td>I think Arial was set as default as it's requi...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text is describing a problem (a user error...</td>
      <td>Yes, this describes a solution (a mechanism). ...</td>
      <td>The link between the setting and the problem i...</td>
      <td>This mechanism (the default settings) aligns w...</td>
      <td>True</td>
      <td>The text identifies a problem (user error in m...</td>
      <td>0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>False</td>
      <td>incorrect</td>
      <td>The executor incorrectly classified a user err...</td>
    </tr>
    <tr>
      <th>372</th>
      <td>722</td>
      <td>usability</td>
      <td>jsroot: better grouping for context menus; Whi...</td>
      <td>\r\n### Data for Evaluation\r\n\r\n**1. Qualit...</td>
      <td>The text discusses a feature (jsroot) and its ...</td>
      <td>Yes, it describes an architectural mechanism b...</td>
      <td>The link between the implementation of jsroot ...</td>
      <td>This matches the definition of usability as it...</td>
      <td>True</td>
      <td>The analysis concludes that the text describes...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>False</td>
      <td>incorrect</td>
      <td>The executor's reasoning was invalid because i...</td>
    </tr>
  </tbody>
</table>
<p>81 rows × 15 columns</p>
</div>




```python
df[["s1_true_positive", "s1_v_evaluation"]].value_counts()
```




    s1_true_positive  s1_v_evaluation
    True              correct            157
    False             correct            146
    True              incorrect           47
    False             incorrect           34
    Name: count, dtype: int64




```python
df.dmitry_approves = df.dmitry_approves.astype(bool)
```


```python
df.groupby("dmitry_approves").size().rename("count")
```




    dmitry_approves
    False     63
    True     321
    Name: count, dtype: int64




```python
df.groupby("dmitry_approves").size().rename("percentage") / df.shape[0] * 100
```




    dmitry_approves
    False    16.40625
    True     83.59375
    Name: percentage, dtype: float64




```python
df.groupby(["dmitry_approves", "s1_v_evaluation"]).size().rename("count")
```




    dmitry_approves  s1_v_evaluation
    False            correct             19
                     incorrect           44
    True             correct            284
                     incorrect           37
    Name: count, dtype: int64



# S2


```python
df = CSVDFHandler().read_df(AbsDirPath.SAMPLES_VERIFIED / "s2_manual_ver.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>sentence</th>
      <th>s2_prompt</th>
      <th>s2_analysis_summary</th>
      <th>s2_architectural_signal</th>
      <th>s2_exclusionary_signal</th>
      <th>s2_final_logic</th>
      <th>s2_related_to_arch</th>
      <th>dmitry_approves</th>
      <th>s2_v_prompt</th>
      <th>s2_v_ground_truth_classification</th>
      <th>s2_v_ground_truth_rule</th>
      <th>s2_v_evaluation</th>
      <th>s2_v_reasoning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>226</td>
      <td>The type MachO::macho_load_command is defined ...</td>
      <td>### Data for Evaluation\r\n\r\n**Content to An...</td>
      <td>The text discusses the definition of a specifi...</td>
      <td>TRUE</td>
      <td>No exclusionary criteria apply</td>
      <td>TRUE</td>
      <td>True</td>
      <td>0.0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>True</td>
      <td>A3: Cross-Cutting Concerns &amp; Non-Functional Re...</td>
      <td>correct</td>
      <td>The text describes a system-wide structure for...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88</td>
      <td>My HPC team helped me with this error, just th...</td>
      <td>### Data for Evaluation\r\n\r\n**Content to An...</td>
      <td>The text discusses a specific error related to...</td>
      <td>Yes</td>
      <td>No</td>
      <td>TRUE</td>
      <td>True</td>
      <td>1.0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>True</td>
      <td>A3: Cross-Cutting Concerns &amp; Non-Functional Re...</td>
      <td>correct</td>
      <td>The text describes a system-wide constraint on...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>151</td>
      <td>Yes, i'll add a test for this. . Yes, it uses ...</td>
      <td>### Data for Evaluation\r\n\r\n**Content to An...</td>
      <td>The text discusses adding a test for fold chan...</td>
      <td>Yes, i'll add a test for fold changes using th...</td>
      <td>No exclusionary signals found</td>
      <td>TRUE</td>
      <td>True</td>
      <td>1.0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>True</td>
      <td>A3: Cross-Cutting Concerns &amp; Non-Functional Re...</td>
      <td>correct</td>
      <td>The text discusses the use of a specific formu...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>124</td>
      <td>Thank you very much for pointing me to this! W...</td>
      <td>### Data for Evaluation\r\n\r\n**Content to An...</td>
      <td>The text discusses a significant architectural...</td>
      <td>TRUE</td>
      <td>No exclusionary criteria apply because the tex...</td>
      <td>The text identifies strong architectural signa...</td>
      <td>True</td>
      <td>0.0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>True</td>
      <td>E3: Trivial Setup</td>
      <td>incorrect</td>
      <td>The AI's classification is incorrect because t...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>s should default to using the collapseRequests...</td>
      <td>### Data for Evaluation\r\n\r\n**Content to An...</td>
      <td>The text discusses a critical architectural de...</td>
      <td>Yes</td>
      <td>No</td>
      <td>TRUE</td>
      <td>True</td>
      <td>1.0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>True</td>
      <td>A3: Cross-Cutting Concerns &amp; Non-Functional Re...</td>
      <td>correct</td>
      <td>The AI's classification is correct because the...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (384, 14)




```python
df.groupby("s2_v_evaluation").size().rename("percentage") / df.shape[0] * 100
```




    s2_v_evaluation
    correct      73.958333
    incorrect    26.041667
    Name: percentage, dtype: float64




```python
df.groupby("s2_v_evaluation").size().rename("count")
```




    s2_v_evaluation
    correct      284
    incorrect    100
    Name: count, dtype: int64




```python
df.dmitry_approves = df.dmitry_approves.astype(bool)
```


```python
df.groupby("dmitry_approves").size().rename("count")
```




    dmitry_approves
    False    100
    True     284
    Name: count, dtype: int64




```python
df.groupby("dmitry_approves").size().rename("percentage") / df.shape[0] * 100
```




    dmitry_approves
    False    26.041667
    True     73.958333
    Name: percentage, dtype: float64




```python
df.groupby(["dmitry_approves", "s2_v_evaluation"]).size().rename("count")
```




    dmitry_approves  s2_v_evaluation
    False            correct             33
                     incorrect           67
    True             correct            251
                     incorrect           33
    Name: count, dtype: int64



# S3


```python
df = CSVDFHandler().read_df(AbsDirPath.SAMPLES_VERIFIED / "s3_manual_ver.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>qa</th>
      <th>sentence</th>
      <th>s3_prompt</th>
      <th>s3_architectural_activity_extraction</th>
      <th>s3_core_concept_analysis</th>
      <th>s3_is_tactic_relevant</th>
      <th>s3_relevance_reason</th>
      <th>s3_tactic_evaluation</th>
      <th>s3_selected_tactic</th>
      <th>s3_tactic_desc</th>
      <th>s3_justification</th>
      <th>dmitry_approves</th>
      <th>s3_v_prompt</th>
      <th>s3_v_evaluation</th>
      <th>s3_v_reasoning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27922</td>
      <td>testability</td>
      <td>rder, our approach is pinning it temporarily u...</td>
      <td>\nBased on the rules provided in the system pr...</td>
      <td>rder, our approach is pinning it temporarily u...</td>
      <td>The primary architectural concept identified i...</td>
      <td>True</td>
      <td>The concept identified is central to understan...</td>
      <td>TRUE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>The concept identified in this text is version...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>correct</td>
      <td>The executor's claim that the discussion about...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4089</td>
      <td>modifiability</td>
      <td>Hi @moldach . Thank you for reporting this! Fo...</td>
      <td>\r\nBased on the rules provided in the system ...</td>
      <td>I drafted a few new sections in the Quick Star...</td>
      <td>Split Module</td>
      <td>True</td>
      <td>The extracted sentence discusses a specific ac...</td>
      <td>TRUE</td>
      <td>Split Module</td>
      <td>Refining a module into several smaller modules...</td>
      <td>The extracted sentence clearly indicates that ...</td>
      <td>0</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>correct</td>
      <td>The executor's claim in Step 1 is that they ar...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7174</td>
      <td>deployability</td>
      <td>API extraction should not generate an actual p...</td>
      <td>\r\nBased on the rules provided in the system ...</td>
      <td>API extraction should not generate an actual p...</td>
      <td>The primary architectural concept is preventin...</td>
      <td>True</td>
      <td>The extracted activity prevents generating a p...</td>
      <td>split module</td>
      <td>Split Module</td>
      <td>Refining a module into several smaller modules...</td>
      <td>The selected tactic 'split module' best matche...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>correct</td>
      <td>The executor's claim that the discussion on de...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5247</td>
      <td>usability</td>
      <td>Hi Masaru,. I've filed an internal issue to tr...</td>
      <td>\r\nBased on the rules provided in the system ...</td>
      <td>Hi Masaru,. I've filed an internal issue to tr...</td>
      <td>We'll keep usability for beginners in mind for...</td>
      <td>True</td>
      <td>The text discusses a design decision related t...</td>
      <td>Maintain system models</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>The selected tactic 'Maintain system models' d...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>correct</td>
      <td>The executor's claim that the text describes a...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2321</td>
      <td>availability</td>
      <td>rent from the make_examples command you posted...</td>
      <td>\r\nBased on the rules provided in the system ...</td>
      <td>I tested with `sudo docker run -it gcr.io/deep...</td>
      <td>The primary architectural concept identified i...</td>
      <td>True</td>
      <td>The extracted sentence discusses a concrete te...</td>
      <td>TRUE</td>
      <td>Ping/Echo</td>
      <td>An asynchronous request/response message pair ...</td>
      <td>The extracted sentence discusses the use of Do...</td>
      <td>1</td>
      <td>Now, perform your your audit based on the data...</td>
      <td>correct</td>
      <td>The executor's claim that the discussion on Do...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (384, 16)




```python
df.groupby("s3_v_evaluation").size().rename("percentage") / df.shape[0] * 100
```




    s3_v_evaluation
    correct      82.291667
    incorrect    17.447917
    Name: percentage, dtype: float64




```python
df.groupby("s3_v_evaluation").size().rename("count")
```




    s3_v_evaluation
    correct      316
    incorrect     67
    Name: count, dtype: int64




```python
df.dmitry_approves = df.dmitry_approves.astype(bool)
```


```python
df.groupby("dmitry_approves").size().rename("count")
```




    dmitry_approves
    False     71
    True     313
    Name: count, dtype: int64




```python
df.groupby("dmitry_approves").size().rename("percentage") / df.shape[0] * 100
```




    dmitry_approves
    False    18.489583
    True     81.510417
    Name: percentage, dtype: float64




```python
df.groupby(["dmitry_approves", "s3_v_evaluation"]).size().rename("count")
```




    dmitry_approves  s3_v_evaluation
    False            correct             29
                     incorrect           41
    True             correct            287
                     incorrect           26
    Name: count, dtype: int64



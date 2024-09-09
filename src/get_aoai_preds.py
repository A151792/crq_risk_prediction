from pgr_snowflake.connect import connect_user_via_oauth_noninteractive
from dotenv import load_dotenv
import os
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from openai import AzureOpenAI
import re
from snowflake.connector.pandas_tools import write_pandas


load_dotenv()

conn = connect_user_via_oauth_noninteractive(
        user_upn=os.environ['SNOWFLAKE_UPN'],
        user_pwd=os.environ['SNOWFLAKE_PWD'],
        sf_user='a151792',
        warehouse='IT_DEVELOPER')
        # database='ITSM',
        # schema='PUBLISHED_RESTRICTED')

cur = conn.cursor()

cur.execute("""select 
    t1.CHANGEID,    
    t1.ChangeManagerPartyLanID,
    t1.ChangeManagerPartyName,
    t1.ChangeStatusName,
    t1.ChangeSubmitterPartyLanID,
    t1.ChangeProdCatTier1Name,
    t1.ChangeProdCatTier2Name,
    t1.ChangeProdCatTier3Name,
    t1.ChangeProdName,
    t1.ChangeReasonTypeName,
    t1.ChangeScheduledStartDateTime,
    year(t1.ChangeScheduledStartDateTime) as year,
    t1.ChangeScheduledEndDateTime,
    t1.ChangeSubmittedDateTime,
    t1.ChangeSubmitterPartyName,
    t1.ChangeOpCatTier1Name,
    t1.ChangeOpCatTier2Name,
    t1.ChangeOpCatTier3Name,
    t1.ChangeRiskLevelName,
    t2.ChangeSummaryText,
    t2.ChangeNoteText,
    t1.ChangeManagerSupportGroupName
from
    ITSM.PUBLISHED_RESTRICTED.vw_FactITSMChangeMeasure t1
join 
    ITSM.PUBLISHED_RESTRICTED.vw_FactITSMChangeMeasureTextColumns t2 
    on t1.CHANGEID = t2.CHANGEID
where 
    t1.ChangeManagerSupportGroupName = 'Change Management'
    and NOT EXISTS (
        select 1
        from "DSC_SANDBOX"."USER_MANAGED_RESTRICTED"."CRQ_DATA_SF_PIPE" cd
        where cd.INFRASTRUCTURE_CHANGE_ID = t1.CHANGEID
    )
    and year=2024
order by 
    t1.ChangeScheduledStartDateTime desc
""")

df = cur.fetch_pandas_all()

# get rubric
rules_df = pd.read_csv("../data/rubrics/crq_rubric_v1.csv")
rules_df = rules_df.dropna(subset=['Weight (1-5)'])
rules_df['Risk_Level'] = rules_df['Risk Level: 1-3 (Low,Mod,High)'].astype('str')
rules_df['Weight'] = rules_df['Weight (1-5)'].astype('str')
rules_df['llm_rule'] = rules_df['Rule']+':'+rules_df['Risk_Level']+':'+rules_df['Weight']
rubric = ','.join(rules_df['llm_rule'])

if df.columns.duplicated().any():
    raise ValueError("DataFrame contains duplicate column names.")

df.insert(1, 'LLM_input','')
# print(df.columns)
for col in df.columns[2:]:
    df['LLM_input'] = df['LLM_input'] + str(col) + ':' + df[col].astype('str') + ','

def generate_prompt(user_input):
    prompt = f"Translate the following English text to French: {user_input}"
    return prompt

def get_pred(input):

    prompt = """
    You are a change management expert who deals with infrastructure changes and managing change requests in an enterprise IT environment.
            Change requests will hereafter be referred to as "CRQ".
            Given this new CRQ information delimited by tiple backticks ```{new_crq}```,
            you should infer the risk of the new by giving a risk score of low, medium, high.
            Use the information contained in the crq to help determine its risk.
            Provide an explanation of why you gave the level of risk score and how you came to that score.
    
            Each field of the CRQ is represented as a field:value pair and each pair is delimited by a comma.
            The following fields have values that map to the true values.
            Each map below is listed as field:value - true value
            
            
            Below is a list of rules delimted by triple backticks created by a change management expert who deals with CRQs.
            Each rule has an associated risk level (1,2,3), and an associated weight (1,2,3,4,5).
            Each rule is in the format of `rule:risk level (low, moderate, high):weight (1-5)`.
            Each rule is delimited by a comma.
            You should use these rules as supplemental information when making your risk analyses,
            but do not rely solely on them to determine a risk score.
    
            The implementation plan is only valid if it is in the notes or there is a sharepoint link to the implementation plan.
            
            The verification plan is only valid if it is in the notes or there is a sharepoint link to the verification plan.
            
            The backout plan is only valid if it is in the notes or there is a sharepoint link to the backout plan.    
    
            If the implementation, verification, or backout plan do not have a description or a sharepoint link, they should be considered absent.
    
            Do not infer risk by negating the rules in the rubric. Do not give reasoning based on a rule NOT applying to a CRQ.
    
            The technology-based analyis and rules-based analysis sections of the output should be numbered lists.
    
            In the rules-based analysis section, list the rule referenced for each numbered item in parentheses at the end.
    
            Rules: ```{rubric}```
    
            Do not include any asterisks in your response. Your response should be concise and not verbose. 

            Your response should be in the following format:
            
            Risk score:
    
            Summary of work entailed:
    
            Technology-based analysis:
    
            Rules-based analysis:
    
            Summary of risk:
    """

    client = AzureOpenAI()

    settings = {
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 1000,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt.format(new_crq=input, rubric=rubric),
            },
        ], **settings)
    
    return response.choices[0].message.content

# get predictions
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_chunk(df, chunk_number):
    df['LLM_output'] = df['LLM_input'].apply(lambda x: get_pred(x))

chunks = np.array_split(df, len(df))

def extract_sections(text):
    # Check if the input is a string
    if not isinstance(text, str):
        return pd.Series({
            'LLM_SCORE': '',
            'WORK_SUMMARY': '',
            'TECH_ANALYSIS': '',
            'RULES_ANALYSIS': '',
            'RISK_SUMMARY': ''
        })

    sections = {
        'LLM_SCORE': '',
        'WORK_SUMMARY': '',
        'TECH_ANALYSIS': '',
        'RULES_ANALYSIS': '',
        'RISK_SUMMARY': ''
    }

    # Define regex patterns for each section
    patterns = {
        'LLM_SCORE': r'Risk score:\s*(.*?)\s*Summary of work entailed:',
        'WORK_SUMMARY': r'Summary of work entailed:\s*(.*?)\s*Technology-based analysis:',
        'TECH_ANALYSIS': r'Technology-based analysis:\s*(.*?)\s*Rules-based analysis:',
        'RULES_ANALYSIS': r'Rules-based analysis:\s*(.*?)\s*Summary of risk:',
        'RISK_SUMMARY': r'Summary of risk:\s*(.*)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()

    return pd.Series(sections)

conn2 = connect_user_via_oauth_noninteractive(
        user_upn=os.environ['SNOWFLAKE_UPN'],
        user_pwd=os.environ['SNOWFLAKE_PWD'],
        sf_user='a151792',
        warehouse='IT_DEVELOPER',
        database='DSC_SANDBOX',
        schema='USER_MANAGED_RESTRICTED')

for i in range(0, len(chunks), 1):
    print(f"chunk {i}... of {len(chunks)}")
    process_chunk(df=chunks[i], chunk_number=i)
    # Apply the function to the DataFrame and directly assign the result to new columns
    chunks[i][['LLM_SCORE', 'WORK_SUMMARY', 'TECH_ANALYSIS', 'RULES_ANALYSIS', 'RISK_SUMMARY']] = chunks[i]['LLM_output'].apply(extract_sections)
    print(f"writing chunk {i}")
    chunks[i].columns = chunks[i].columns.str.upper()
    chunks[i].index = pd.RangeIndex(start=0, stop=len(chunks[i]), step=1)
    # chunks[i] = chunks[i].drop('LLM_OUTPUT', axis=1)
    # print('after drop')
    # for col in chunks[i].columns: print(col)
    # chunks[i] = chunks[i].reset_index(drop=True)

    # breakpoint()

    chunks[i].to_csv('variant_test.csv')

    chunks[i]['PROMPT_VERSION'] = 1
    chunks[i]['RUBRIC_VERSION'] = 1

    chunks[i]['CHANGESCHEDULEDSTARTDATETIME'] = chunks[i]['CHANGESCHEDULEDSTARTDATETIME'].dt.date

    chunks[i]['CHANGESCHEDULEDENDDATETIME'] = pd.to_datetime(chunks[i]['CHANGESCHEDULEDENDDATETIME'])
    chunks[i]['CHANGESCHEDULEDENDDATETIME'] = chunks[i]['CHANGESCHEDULEDENDDATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')

    chunks[i]['CHANGESUBMITTEDDATETIME'] = pd.to_datetime(chunks[i]['CHANGESUBMITTEDDATETIME'])

    chunks[i]['CHANGESUBMITTEDDATETIME'] = chunks[i]['CHANGESUBMITTEDDATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')


    # breakpoint()

    try:
        success, nchunks, nrows, _ = write_pandas(conn2, chunks[i], "CRQ_DATA_SF_PIPE", index=False)
        # print(success)
    except Exception as e:
        print(e) 
        continue
    # print(success)
    # chunks[i].to_sql('"DSC_SANDBOX"."USER_MANAGED_RESTRICTED"."CRQ_DATA"', conn, if_exists='append', index=False)

import os
from langchain_community.graphs import Neo4jGraph
from langchain.chat_models import init_chat_model
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def query_neo4j(query: str) -> str:
    
    graph = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database="neo4j"
    ) 
    
    CYPHER_GENERATION_TEMPLATE = """
    Task: Generate Cypher query to query the Neo4j Healthcare Analytics graph database.
     **Graph Database:** The graph database contains information about drugs, manufacturers, therapies, reactions, and patient cases. It is a Neo4j graph database.
     **Data Model:** The graph database schema is as follows:
        **Nodes:**
        - ´Case´ nodes (properties: ´primaryId´, ´age´, ´ageUnit´, ´eventDate´, ´gender´, ´reportDate´, ´reporterOccupation´) represent patient cases
        - ´Drug´ nodes (properties: ´name´, ´primarySubstabce´) represent drugs. 
        - ´Manufacturer´ nodes (properties: ´manufacturerName´) represent drug manufacturers
        - ´Therapy´ nodes (properties: ´primaryId´) represent therapies
        - ´Reaction´ nodes (properties: ´description´) represent adverse reactions
        - ´Outcome´ nodes (properties: ´outcome´, ´code´) represent outcomes of cases
        - ´ReportSource´ nodes (properties: ´code´, ´name´) represent sources of reports
        - ´AgeGroup´ nodes (properties: ´ageGroup´) represent age groups of patients
        **Relationships:**
        - ´FALLS_UNDER´ relationship connects ´Case´ nodes to ´AgeGroup´ nodes, indicating the age group of the patient in the case
        - ´HAS_REACTION´ relationship connects ´Case´ nodes to ´Reaction´ nodes, indicating the reactions experienced in the case
        - ´RESULTED_IN´ relationship connects ´Case´ nodes to ´Outcome´ nodes, indicating the outcome of the case
        - ´REPORTED_BY´ relationship connects ´Case´ nodes to ´ReportSource´ nodes, indicating the source of the report
        - ´REGISTERED´ relationship connects ´Case´ nodes to ´Manufacturer´ nodes, indicating the manufacturer that was registered in the case
        - ´PRESCRIBED´ relationship connects ´Therapy´ nodes to ´Drug´ nodes, indicating the drug prescribed in the therapy
        - ´RECEIVED´ relationship connects ´Case´ nodes to ´Therapy´ nodes, indicating the therapy received in the case
        - ´IS_CONCOMITANT´ relationship (properties: ´drugSequence´, ´route´, ´indication´, ´doseAmount´, ´doseUnit´) connects ´Case´ nodes to ´Drug´ nodes, indicating concomitant drugs in the case
        - ´IS_INTERACTING´ relationship (properties: ´drugSequence´, ´route´, ´indication´, ´doseAmount´, ´doseUnit´) connects ´Case´ nodes to ´Drug´ nodes, indicating interactions between drugs in the case
        - ´IS_PRIMARY_SUSPECT´ relationship (properties: ´drugSequence´, ´route´, ´indication´, ´doseAmount´, ´doseUnit´) connects ´Case´ nodes to ´Drug´ nodes, indicating the primary suspect drug in the case
        - ´IS_SECONDARY_SUSPECT´ relationship (properties: ´drugSequence´, ´route´, ´indication´, ´doseAmount´, ´doseUnit´) connects ´Case´ nodes to ´Drug´ nodes, indicating the secondary suspect drug in the case
    
     **Query Generation:** Generate a Cypher query to answer the question based on the graph database schema. The ´name´ and ´primarySubstance´ properties of the ´Drug´ nodes and the ´manufacturerName´ property in ´Manufacturer´ nodes have to be converted to upper case in the generated Cypher query.
     **Question:** {query}          
    ...
    """
    CYPHER_GENERATION_PROMPT = PromptTemplate(input_variables=["schema", "query"], template=CYPHER_GENERATION_TEMPLATE)

    llm = init_chat_model(
        "gemini-2.0-flash", 
        model_provider="google_genai",
        temperature=0,
        max_tokens=1024
    )

    graph_chain = GraphCypherQAChain.from_llm(
        llm=llm, 
        graph=graph, 
        cypher_prompt=CYPHER_GENERATION_PROMPT, 
        verbose=True,
        allow_dangerous_requests=True
    )
    return graph_chain.invoke(query)

   
   
   
   
   
#**Example Questions and Cypher Query:** 
        # - "What are the top 5 side effects reported?"
        #     Cypher Query: MATCH (c:Case)-[:HAS_REACTION]->(r:Reaction) RETURN r.description, count(c) ORDER BY count(c) DESC LIMIT 5;
        # - "What are the top 5 drugs reported with side effects? Get drugs along with their side effects."
        #     Cypher Query: MATCH (c:Case)-[:IS_PRIMARY_SUSPECT]->(d:Drug) MATCH (c)-[:HAS_REACTION]->(r:Reaction) WITH d.name as drugName, collect(r.description) as sideEffects, count(r.description) as totalSideEffects RETURN drugName, sideEffects[0..5] as sideEffects, totalSideEffects ORDER BY totalSideEffects DESC LIMIT 5;
        # - "What are the manufacturing companies which have most drugs which reported side effects?"
        #     Cypher Query: MATCH (m:Manufacturer)-[:REGISTERED]->(c)-[:HAS_REACTION]->(r) RETURN m.manufacturerName as company, count(distinct r) as numberOfSideEffects ORDER BY numberOfSideEffects DESC LIMIT 5;    
        # - "What are the top 5 drugs from a particular company with side effects? What are the side effects from those drugs?"
        #     Cypher Query: MATCH (m:Manufacturer {manufacturerName: "NOVARTIS"})-[:REGISTERED]->(c) MATCH (r:Reaction)<--(c)-[:IS_PRIMARY_SUSPECT]->(d) WITH d.name as drug,collect(distinct r.description) AS reactions, count(distinct r) as totalReactions RETURN drug, reactions[0..5] as sideEffects, totalReactions ORDER BY totalReactions DESC LIMIT 5;
        # - "What are the top 3 drugs which are reported directly by consumers for the side effects?"
        #     Cypher Query: MATCH (c:Case)-[:REPORTED_BY]->(rpsr:ReportSource {name: "Consumer"}) MATCH (c)-[:IS_PRIMARY_SUSPECT]->(d) MATCH (c)-[:HAS_REACTION]->(r) WITH rpsr.name as reporter, d.name as drug, collect(distinct r.description) as sideEffects, count(distinct r) as total RETURN drug, reporter, sideEffects[0..5] as sideEffects ORDER BY total desc LIMIT 3;
        # - "What are the top 2 drugs whose side effects resulted in Death of patients as an outcome?"
        #     Cypher Query: MATCH (c:Case)-[:RESULTED_IN]->(o:Outcome {outcome:"Death"}) MATCH (c)-[:IS_PRIMARY_SUSPECT]->(d) MATCH (c)-[:HAS_REACTION]->(r) WITH d.name as drug, collect(distinct r.description) as sideEffects, o.outcome as outcome, count(distinct c) as cases RETURN drug, sideEffects[0..5] as sideEffects, outcome, cases ORDER BY cases DESC LIMIT 2;
        # - "Show top 10 drug combinations which have most side effects when consumed together."
        #     Cypher Query: MATCH (c:Case)-[:IS_PRIMARY_SUSPECT]->(d1) MATCH (c:Case)-[:IS_SECONDARY_SUSPECT]->(d2) MATCH (c)-[:HAS_REACTION]->(r) MATCH (c)-[:RESULTED_IN]->(o) WHERE d1<>d2 WITH d1.name as primaryDrug, d2.name as secondaryDrug, collect(r.description) as sideEffects, count(r.description) as totalSideEffects, collect(o.outcome) as outcomes RETURN primaryDrug, secondaryDrug, sideEffects[0..3] as sideEffects, totalSideEffects, outcomes[0] ORDER BY totalSideEffects desc LIMIT 10;
        # - "What is the age group which reported highest side effects, and what are those side effects?"
        #     Cypher Query: MATCH (a:AgeGroup)<-[:FALLS_UNDER]-(c:Case) MATCH (c)-[:HAS_REACTION]->(r) WITH a, collect(r.description) as sideEffects, count(r) as total RETURN a.ageGroup as ageGroupName, sideEffects[0..6] as sideEffects ORDER BY total DESC LIMIT 1;
        # - "What is the age group which has most patients treated with Lyrica?"
        #     Cypher Query: MATCH (c:Case)-[:IS_PRIMARY_SUSPECT]->(d:Drug {name: "LYRICA"})<-[:PRESCRIBED]-(t:Therapy) MATCH (c)-[:FALLS_UNDER]->(a:AgeGroup) WITH a.ageGroup as ageGroupName, count(c) as total RETURN ageGroupName, total ORDER BY total DESC LIMIT 1;
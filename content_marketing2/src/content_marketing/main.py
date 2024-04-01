from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

from crewai import Agent, Crew, Process, Task

search_tool = SerperDevTool()
max_iterations = 3 
llm_model = ChatOpenAI(
                model="crewai-mistral",
                base_url="http://127.0.0.1:11434/v1",
                openai_api_key="NA"
        )

researcher = Agent(
        role='Content Marketing Researcher',
        goal='Uncover groundbreaking technologies and trends in the content marketing industry',
        verbose=True,
        memory=False,
        backstory=(
                "Driven by curiosity, you're at the forefront of"
                "innovation, eager to explore and share knowledge that could change"
                "the world."
        ),
        tools=[search_tool],
        allow_delegation=True,
        llm=llm_model,
        max_iter = max_iterations
)

writer = Agent(
        role='Writer',
        goal='Create compelling articles about content marketing trends and technologies',
        verbose=True,
        memory=False,
        backstory=(
                "With a flair for simplifying complex topics, you craft"
                "engaging narratives that captivate and educate, bringing new"
                "discoveries to light in an accessible manner."
        ),
        tools=[search_tool],
        allow_delegation=False,
        llm=llm_model,
        max_iter = max_iterations
)

# Research task
research_task = Task(
    description=(
        "Identify the next big trend in content marketing."
        "Focus on identifying pros and cons and the overall narrative."
        "Your final report should clearly articulate the key points"
        "its market opportunities, and potential risks."
    ),
    expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
    tools=[search_tool],
    agent=researcher,
)

# Writing task with language model configuration
write_task = Task(
    description=(
        "Compose an insightful article on content marketing."
        "Focus on the latest trends and how it's impacting the industry."
        "This article should be easy to understand, engaging, and positive."
    ),
    expected_output='An engaging article on content marketing trends and advancements formatted as markdown.',
    tools=[search_tool],
    agent=writer,
    async_execution=False,
    output_file='new-blog-post.md'
)

# Forming the tech-focused crew with enhanced configurations
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential  
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'Content Marketing Trends'})

print(result)

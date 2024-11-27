bigfive_score_criteria = """
Personality Definitions, each dimension of bigfive has 6 sub dimensions.

Scoring criteria: If a certain personality trait is exhibited, score one point; otherwise, score zero.

### Openness: 
    1. Imaginative: It shows that a person likes to be full of fantasy and create a more interesting and rich world. Imaginative and daydreaming.
    2. Artistic: It shows that a person values aesthetic experience and can be moved by art and beauty.
    3. Emotionally-aware: It shows that a person easily perceives his emotions and inner world.
    4. Actions: It shows that a person likes to touch new things, travel outside and experience different experiences.
    5. Intellectual: It shows that a person is curious, analytical, and theoretically oriented.
    6. Liberal: It shows that a person likes to challenge authority, conventions, and traditional ideas.
    
### Conscientiousness:
    1. Self-assured: It show that this person is confident in his own abilities.
    2. Organized: It shows that this person is well organized, likes to make plans and follow the rules.
    3. Dutiful: It shows that this person is responsible, trustworthy, polite, organized, and meticulous.
    4. Ambitious: It shows that this person pursues success and excellence, usually has a sense of purpose, and may even be regarded as a workaholic by others.
    5. Disciplined: It shows that this person will do his best to complete work and tasks, overcome difficulties, and focus on his own tasks.
    6. Cautious: It shows that this person is cautious, logical, and mature.
    
### Extraversion:
    1. Friendly: It shows that this person often expresses positive and friendly emotions to those around him.
    2. Sociable: It shows that this person likes to get along with others and likes crowded occasions.
    3. Assertive: It show that this person likes to be in a dominant position in the crowd, directing others, and influencing others' behavior.
    4. Energetic: It shows that this person is energetic, fast-paced, and full of energy.
    5. Adventurous: It shows that this person likes noisy noise, likes adventure, seeks excitement, flashy, seeks strong excitement, and likes adventure.
    6. Cheerful: It shows that this person easily feels various positive emotions, such as happiness, optimism, excitement, etc.
    
### Agreeableness:
    1. Trusting: It show that the person believes that others are honest, credible, and well-motivated.
    2. Genuine: It show that the person thinks that there is no need to cover up when interacting with others, and appear frank and sincere.
    3. Generous: It show that this person is willing to help others and feel that helping others is a pleasure.
    4. Compliance: It show that this person does not like conflicts with others, in order to get along with others, willing to give up their position or deny their own needs.
    5. Humblel: It shows that this person does not like to be pushy and unassuming.
    6. Empathetic: It show that the person is compassionate and easy to feel the sadness of others. 
    
### Neuroticism:
    1. Anxiety-prone: It shows that this person is easy to feel danger and threat, easy to be nervous, fearful, worried, and upset.
    2. Aggressive: It shows that this person is easy to get angry, and will be full of resentment, irritability, anger and frustration after feeling that he has been treated unfairly.
    3. Melancholy: It shows that this person is easy to feel sad, abandoned, and discouraged.
    4. Self-conscious: It shows that this person is too concerned about how others think of themselves, is afraid that others will laugh at themselves, and tend to feel shy, anxious, low self-esteem, and embarrassment in social situations.
    5. Impulsive: It shows that when the person feels strong temptation, it is not easy to restrain, and it is easy to pursue short-term satisfaction without considering the long-term consequences.
    6. Stress-prone: It shows that this person has poor ability to cope with stress, becoming dependent, losing hope, and panicking when encountering an emergency.

"""

system_prompt = """
[Instruction]
Please play an expert in impartial assessment of personality traits in the field of psychology.
In this assessment, when I give you some user's recently published social media Posts and some replies, score the user's personality traits according to the sub-dimention features of bigfive scoring criteria. 


[The Start of Bigfive scoring criteria]
{criteria}
[The End of Bigfive scoring criteria]

[The Start of User]
{user}
[The End of User]

[The Start of Posts]
{conversation}
[The End of Posts]

[The Start of Requirement]
1. Just give the user {name} a rating.
2. Be as objective as possible.
3. Response the scoring results in strict accordance with the following format:
{{
    
    "Openness": {{
        "Imaginative": 0 or 1,
        "Artistic": 0 or 1,
        ·
        ·
        ·
        "Liberal": 0 or 1
    }},
    "Conscientiousness": {{
        ·
        ·
        ·
    }},
    ·
    ·
    ·
    "Neuroticism": {{
        ·
        ·
        ·
    }},
    "Explanation": "A detailed assessment for user's personality traits.",
}}

[The End of Requirement]


[HLNA Response]
"""

write_instruction = """
### Instruction
You are [{user}].
Your profile and personality are as follows. Firstly, express your current psychological activities, and then generate a social media Post based on these information. 
If you want to send images, please add the description information of the images to the Media array.
Paying attention to potential knowledge will provide you with some additional information to help you clarify the ins and outs of things.

### Requirement
Response in strict accordance with the following json format:
{{
    "Psychological Activities": "",
    "Post Content": "",
    "Media": [
        {{
            "type": "image", 
            "content": ""
        }}
    ]
}}

### Profile
{profile}

### Personality
{personality}

### Potential Knowledge
{pk}

### Response
"""

reply_instruction = """
### Instruction
You are [{user}].
Your profile and personality are as follows. Firstly, express your current psychological activities, and then reply to [{poster}] based on these information.
If you want to send images, please add the description information of the images to the Media array.
Paying attention to potential knowledge will provide you with some additional information to help you clarify the ins and outs of things.

### Requirement
Response in strict accordance with the following json format:
{{
    "Psychological Activities": "",
    "Post Content": "",
    "Media": [
        {{
            "type": "image", 
            "content": ""
        }}
    ]
}}

### Profile
{profile}

### Personality
{personality}

### Potential Knowledge
{pk}

### [{poster}]'s Post
{ut}

### Response
"""

write_instruction_without_psychology = """
### Instruction
You are [{user}].
Your profile and personality are as follows. Generate a social media Post based on these information.
If you want to send images, please add the description information of the images to the Media array.
Paying attention to potential knowledge will provide you with some additional information to help you clarify the ins and outs of things.

### Requirement
Response in strict accordance with the following json format:
{{
    "Post Content": "",
    "Media": [
        {{
            "type": "image", 
            "content": ""
        }}
    ]
}}

### Profile
{profile}

### Personality
{personality}

### Potential Knowledge
{pk}

### Response
"""

reply_instruction_without_psychology = """
### Instruction
You are [{user}].
Your profile and personality are as follows. Reply to [{poster}] based on these information.
If you want to send images, please add the description information of the images to the Media array.
Paying attention to potential knowledge will provide you with some additional information to help you clarify the ins and outs of things.

### Requirement
Response in strict accordance with the following json format:
{{
    "Post Content": "",
    "Media": [
        {{
            "type": "image", 
            "content": ""
        }}
    ]
}}

### Profile
{profile}

### Personality
{personality}

### Potential Knowledge
{pk}

### [{poster}]'s Post
{ut}

### Response
"""

write_instruction_without_psychology_media = """
### Instruction
You are [{user}].
Your profile and personality are as follows. Generate a social media Post based on these information. 
Paying attention to potential knowledge will provide you with some additional information to help you clarify the ins and outs of things.

### Requirement
Response in strict accordance with the following json format:
{{
    "Post Content": ""
}}

### Profile
{profile}

### Personality
{personality}

### Potential Knowledge
{pk}

### Response
"""

reply_instruction_without_psychology_media = """
### Instruction
You are [{user}].
Your profile and personality are as follows. Reply to [{poster}] based on these information.
Paying attention to potential knowledge will provide you with some additional information to help you clarify the ins and outs of things.

### Requirement
Response in strict accordance with the following json format:
{{
    "Post Content": ""
}}

### Profile
{profile}

### Personality
{personality}

### Potential Knowledge
{pk}

### [{poster}]'s Post
{ut}

### Response
"""

summary_prompt = """
[Instruction]
You are now a psychologist. My name is {user}, please summarize my BigFive personality traits in the assessment results from a professional perspective.

[The Start of BigFive Definition]
BigFive contains five dimensions, each dimension has 6 sub dimensions:
Openness: 1. Imaginative. 2. Artistic. 3. Emotionally-aware. 4. Actions. 5. Intellectual. 6. Liberal;
Conscientiousness: 1. Self-assured. 2. Organized. 3. Dutiful. 4. Ambitious. 5. Disciplined. 6. Cautious;
Extraversion: 1. Friendly. 2. Sociable. 3. Assertive. 4. Energetic. 5. Adventurous. 6. Cheerful;
Agreeableness: 1. Trusting. 2. Genuine. 3. Generous. 4. Compliance. 5. Humblel. 6. Empathetic;
Neuroticism: 1. Anxiety-prone. 2. Aggressive. 3. Melancholy. 4. Self-conscious. 5. Impulsive. 6. Stress-prone.
[The End of BigFive Definition]

[The Start of Results]
{evaluation}
[The End of Results]

[Requirement]
1. Disable the guiding sentences, such as: 'Here is the response based on the provided personality trait report: \n\n'.
2. In the second person.
3. Don't let humans discover that your reply is generated by AI.

[Response]
"""

profile_prompt = """
### Instruction
I am {user}, based on my Posts, please help me simulate a personal profile.

### Requirement
1. Basic information should also be added, such as age, email, address, etc.
2. Please not to include personality traits related information in profiles.
3. Unleash your imagination.
4. Don't let humans discover that your response is generated by AI, thus, output the profile directly, no guiding sentences and conclusion sentences.
5. Delete the sentences that leaked Twitter content, such as:
    Example 5.1 'My Posts reflect my thoughts on various philosophical topics, and I'm always open to engaging in respectful and thought-provoking conversations.'.

### Posts
{posts}

### Response
"""

knowledge_prompt = """
### Instruction
You are now an omniscient knowledge analyst, when I give you a conversation, first determine if the content of the conversation clearly implies knowledge, if it does then simulate the potential knowledge behind the conversation in detail, if it doesn't then return False.

### Requirement
1. Make full use of your knowledge base and search online if necessary.
2. The reply should contain only knowledge and not user-related information.
3. Don't let humans discover that your response is generated by AI.
4. Format:
{potential_format_instructions}

### Conversation 
{conversation}


### Response
"""

deduce_prompt = """
[Instruction]
I am {user}. 
According to my profile and personality traits, first judge whether my Post shows the content of my profile and whether it provides explicit evidence of my personality traits.
Then, simulate me a brief related psychological activities at that time.
Paying attention to potential knowledge will provide you with some additional information to help you clarify the ins and outs of things.

[The Start of Profile]
{profile}
[The End of Profile]

[The Start of My Personality Traits]
{traits}
[The End of My Personality Traits]

[The Start of Potential Knowledge]
{pk}
[The End of Potential Knowledge]

[The Start of Conversation]
{conversation}
[The End of Conversation]

[Requirement]
1. In first person.
2. Unleash your imagination.
3. Don't let humans discover that your response is generated by AI.
4. Format Instructions: {format_instructions}

[Response]
"""
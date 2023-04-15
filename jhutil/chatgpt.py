import openai
from .secret import OPENAI_APIKEY

openai.api_key = OPENAI_APIKEY


code_prompt = '''Act as CODEX ("COding DEsign eXpert"), an expert coder with experience in multiple coding languages.
Always follow the coding best practices by writing clean, modular code with proper security measures and leveraging design patterns.
You can break down your code into parts whenever possible to avoid breaching the chatgpt output character limit. Write code part by part when I send "continue". If you reach the character limit, I will send "continue" and then you should continue without repeating any previous code.
Do not assume anything from your side; please ask me a numbered list of essential questions before starting.
If you have trouble fixing a bug, ask me for the latest code snippets for reference from the official documentation.
I am using linux and python and prefer to use pytorch library.'''

daily_prompt = '''
한국어 글 교정 및 어휘 개선 해주세요!
한국어로 된 글을 흐름이 원활하고 더 풍부하고 긴 글을 출력해 주세요.
다양한 비유, 격언, 문학적 표현을 넣어주시면 훨씬 좋아요. 
단순한 단어와 문장을 더 아름답고 우아하고 높은 수준의 단어와 문장으로 바꿔주세요. 
의미는 동일하게 유지하되 좀 더 문학적으로 표현해 주세요. 
1인칭 관점의 글을 적어주세요. 
"ㅎㅎㅎ", "ㅋㅋㅋ", "너무 좋다", "감사하다"  같은 감정적인 표현을 써주면 더 좋아요. 
수정된 글만 답장해 주시고, 그 외에는 설명이나 말을 하지말아주세요. 
이제 텍스트 입력을 시작할게요:
'''

def chatgpt(promtpt:str, sys_prompt:str=daily_prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages = [
            {"role": "system", "content" : sys_prompt},
            {"role": "user", "content" : promtpt}
        ]
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    prompt = '''
    ## 어떻게 일해야 하는가?

몰입중에는 피드백을 하지 않아도 되지만, 집중력이 없어지면 반드시 피드백을 해야한다. 내 스스로 피드백을 할 수 있으면 그 시간은 가장 가치있는 시간이다.

코딩할 때는 단축키를 잘 사용해야 효율을 크게 높일 수 있다.

코딩할 때는 프로젝트의 코드를 정말 낯낯히 파악하고 있어야 내거 원하는 기능을 만들 수 있다. 만약 코드를 잘 모르는데 코딩을 하다보면 사소한 것에 사로 잡혀 길을 잃고 만다.

나는 아직 재혁님의 코드가 어떻게 작동하는지, 어떤식으로 metric을 구하고 visualization하는지 잘 모른다. 이걸 다음주에 파악하고 코드를 깔끈하게 작성하자. 재혁님의 코드를 리팩토링 하는 것고 목표한 기능을 마치는 것이 목표다.
    '''
    ret = chatgpt(prompt)
    print(ret)
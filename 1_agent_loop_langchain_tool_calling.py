from dotenv import load_dotenv

import os
load_dotenv()

from langchain.chat_models import init_chat_model

from langchain.tools import tool

from langchain_core.messages import HumanMessage,SystemMessage, ToolMessage

from langsmith import traceable

MAX_ITERATIONS=10
MODEL ="llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@tool
def get_product_price(product:str)->float:
    """Look up the price of a product in the catalog"""
    print(f"Looking up the price of {product}" )
    prices={
        "apple":100,
        "banana":700,
        "cherry":900,
    }
    return prices.get(product, 0)

@tool
def apply_discount(price:float, discount:str)->float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: Bronze, Silver, Gold"""
    print(f"Applying discount tier to {price} discount tier:{discount}" )
    discounts={
        "Bronze":5,
        "Silver":12,
        "Gold":23,
    }
    d=discounts.get(discount,0)  
    return round(price * (1 - d/100),2)

#Agent loop
@traceable(name='Langchain agent loop')
def run_agent(question:str):
    tools=[get_product_price,apply_discount]
    tools_dict={t.name:t for t in tools}
    llm=init_chat_model(f"groq:{MODEL}",temperature=0,api_key=GROQ_API_KEY)
    llm_with_tools=llm.bind_tools(tools)
    print(f"Question: {question}")
    print("="*60)
    messages=[
        
        SystemMessage(
            content=(
                "You are a helpful shopping assistant"
                "You have access to product catalog tool and a discount tool"
                "STRICT RULES-you must follow these exactly"
                "1.Never guess or assume any product price"
                "YOu must call get product price tool first to get the real price of a product"
                '''2.You must call apply discount tool when it is needed to apply a discount to a price only 
                    after you receive the real price from get_product_price.Pass the exact
                    price returned by get_product_price do not pass a made up number'''
                '''3.Never calculate discount by yourself using math.
                    Always use apply_discount tool'''    
                '''4.If the user does not specify a discount tier
                ask them which tier to use -do not Assume one'''     
                
            )
        ),
        HumanMessage(
            content=question
        ),

         ]
    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"---Iteraion{iteration }")
        ai_message=llm_with_tools.invoke(messages)
        tool_calls=ai_message.tool_calls

        #If no tool calls then answer is returned
        if not tool_calls:
            print(f"\nFianl Answer:{ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None    

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result=run_agent("What is the price of a banana after applying the gold discount tier?");  
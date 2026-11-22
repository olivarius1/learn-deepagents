# =============================================================================
# 这是一个「演示用」的小脚本，包含两个可以被 AI Agent 调用的工具函数：
#   1. calculate        —— 安全的四则运算计算器（用 ast 解析，不使用 eval）
#   2. convert_currency —— 按实时汇率做货币换算（内部调用 get_exchange_rate 查网）
# 最后把这两个函数交给 deepagents 创建的 Agent，让它自己决定什么时候调用。
#
# 阅读顺序建议：先看 calculate 上面的长注释，理解 ast 是什么，
# 再看 evaluate 里那个「递归函数」是怎么把表达式算出来的。
# =============================================================================

import httpx

def get_exchange_rate(from_currency: str, to_currency: str = "CNY") -> dict:
    """查询实时汇率（数据来自 open.er-api.com，免费无需 API Key）。

    当用户询问当前汇率、或需要按最新汇率换算金额时使用本工具。
    注意：本工具只返回汇率，不返回新闻或背景资料，那类问题请用 tavily_search。

    Args:
        from_currency: 原币种代码，如 "USD"、"CNY"、"EUR"
        to_currency:   目标币种代码，默认 "CNY"

    Returns:
        形如 {"from": "USD", "to": "CNY", "rate": 6.69, "updated": "..."} 的字典
    """
    src = from_currency.upper()
    dst = to_currency.upper()

    resp = httpx.get(f"https://open.er-api.com/v6/latest/{src}", timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("result") != "success":
        raise ValueError(f"汇率查询失败：{data.get('error-type')}")

    return {
        "from": src,
        "to": dst,
        "rate": data["rates"][dst],          # 1 单位 src 等于多少 dst
        "updated": data.get("time_last_update_utc"),
    }

def calculate(expression: str) -> float:
    """计算一个数学表达式字符串，返回 float 结果。

    参数：
        expression: 例如 "1 + 2 * 3" 或 "-(4 / 2)" 这样的字符串。

    返回：
        计算结果（float）。

    为什么不用 eval()？
        内置的 eval("...") 会把字符串当 Python 代码执行，
        用户如果传入 "__import__('os').system('rm -rf /')" 这类内容就非常危险。
        所以这里改用 ast（抽象语法树）**只解析、不执行**，
        然后我们自己遍历这棵树，只允许白名单里的运算。
    """

    import ast       # 标准库：把源码字符串解析成「语法树」，并提供遍历工具
    import operator  # 标准库：把运算符符号（如 +）对应成可调用的函数

    # ---------------------------------------------------------------
    # 第 0 步：先做两个「防滥用」检查，避免有人用超长表达式拖垮程序
    # ---------------------------------------------------------------
    if len(expression) > 200:
        raise ValueError("表达式过长")

    # ast.parse 负责把 "1 + 2" 这种字符串变成一棵树（Ast 对象）。
    # mode="eval" 表示「按一个表达式来解析」，
    # 若用默认的 "exec" 则会按「一整段代码」解析，两者生成的结构不同。
    tree = ast.parse(expression, mode="eval")

    # ast.walk(tree) 会以「广度优先」的方式把树里所有节点一个个吐出来。
    # sum(1 for _ in ast.walk(tree)) 就是在数一共有多少个节点，
    # 节点越多说明表达式越复杂（例如嵌套了很多括号），这里限制不超过 64 个。
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("表达式过于复杂")

    # ---------------------------------------------------------------
    # 第 1 步：白名单表 —— 只允许这四种二元运算
    #
    # 表里每一项的 key 是一个「类型」（ast.Add 代表加号这个运算符节点），
    # value 是一个「真正能执行计算的函数」，例如 operator.add(1, 2) == 3。
    # 只有当 key 在表里时我们才会去执行，其它运算一律拒绝。
    # ---------------------------------------------------------------
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv}

    # ---------------------------------------------------------------
    # 第 2 步：递归函数 evaluate —— 把树「从叶子往根」算出来
    #
    # 关于 ast 的核心概念（初学者重点）：
    #   * ast.Constant   = 一个「常量」，也就是写在代码里的字面量数字，如 1、2.5
    #   * ast.BinOp      = 一个「二元运算」，例如 a + b，它有 3 个关键属性：
    #                        .left  —— 左边的子表达式（又可能是一棵子树）
    #                        .op    —— 运算符节点，如 ast.Add
    #                        .right —— 右边的子表达式
    #   * ast.UnaryOp    = 一个「一元运算」，例如 -a 或 +a，属性为 .op 和 .operand
    #
    # 举例："-(1 + 2)" 解析出的树大致是：
    #       UnaryOp(op=USub, operand=BinOp(left=Constant(1), op=Add, right=Constant(2)))
    # 于是 evaluate 会这样递归：
    #       evaluate(整个表达式)
    #         -> 是一元负号，先算 operand
    #              -> 是加法，先算 left 再算 right
    #                   -> Constant(1) 直接返回 1.0
    #                   -> Constant(2) 直接返回 2.0
    #              -> 1.0 + 2.0 = 3.0
    #         -> 3.0 取负 = -3.0
    # ---------------------------------------------------------------
    def evaluate(node):
        """递归地计算语法树上的一个节点，返回 float。"""

        # 情况一：常量节点。
        # isinstance(node, ast.Constant) 判断它是不是字面量；
        # type(node.value) in (int, float) 再确认值的类型是数字
        # （因为 Constant 也可能装字符串 "abc" 或 True/False，那些都不允许）。
        # 注意这里用 type(...) in (...) 而不是 isinstance，
        # 是为了严格排除 bool（在 Python 里 True 其实是 int 的子类）。
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)  # 统一转成 float，保证结果类型一致

        # 情况二：二元运算节点（加减乘除）。
        # 先检查 type(node.op) 是否在我们的白名单里，
        # 通过后才分别递归计算左边和右边，最后用对应的函数把两者算起来。
        # 这里的递归是关键：left / right 本身可能又是 BinOp 或 UnaryOp。
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.left), evaluate(node.right))

        # 情况三：一元运算节点，只允许 +x 和 -x。
        # ast.UAdd 是正号（其实什么都不用做），ast.USub 是负号（取相反数）。
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)  # 先算出被作用的那个值
            return value if isinstance(node.op, ast.UAdd) else -value

        # 其它一切节点（函数调用、变量名、位运算、幂运算 ** 等）都不认识，
        # 直接抛错。这就是「白名单」：不认识的，一律拒绝执行。
        raise ValueError("只支持数字、括号和 + - * / 运算")

    # ---------------------------------------------------------------
    # 第 3 步：从树的根开始算
    # tree 是 ast.Expression 对象（mode="eval" 的产物），
    # 真正装表达式的是它的 .body 属性，所以这里传 tree.body。
    # ---------------------------------------------------------------
    return evaluate(tree.body)


def convert_currency(amount: float, from_currency: str, to_currency: str = "CNY") -> dict:
    """按【实时汇率】把一种货币换算成另一种货币。

    参数：
        amount:        金额，例如 100
        from_currency: 原币种代码，如 "USD" / "CNY" / "EUR"
        to_currency:   目标币种代码，默认是 "CNY"

    返回：
        形如 {"from": "USD", "to": "CNY", "rate": 6.69,
              "amount": 669.33, "currency": "CNY"} 的字典。
        把 rate 一起返回，模型才能说明「用的是哪天的什么汇率」。

    说明：汇率由 get_exchange_rate() 从 open.er-api.com 实时获取。
    """
    src = from_currency.upper()
    dst = to_currency.upper()

    # 关键概念：这里拿到的 rate 是「成对汇率」，含义是
    #     1 个 src 能换多少 dst        例如 1 USD = 6.69 CNY
    # 所以金额只需乘一次。
    
    rate = get_exchange_rate(src, dst)["rate"]

    # round(..., 2) 保留两位小数
    return {
        "from": src,
        "to": dst,
        "rate": rate,                      # 把汇率暴露给模型，便于它核对/说明依据
        "amount": round(amount * rate, 2),
        "currency": dst,
    }


# =============================================================================
# 下面是「组装并运行 Agent」的部分。
# 虽然 import 一般习惯写在文件最上面，但 Python 允许写在任意位置，
# 这里为了突出「上面是工具函数、下面是使用示例」才放在中间。
# =============================================================================
import os

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

# 创建一个大模型客户端。
# 这里用的是硅基流动（SiliconFlow）提供的 OpenAI 兼容接口，
# 所以依然用 langchain_openai 的 ChatOpenAI，只是换了 base_url。
model = ChatOpenAI(
    # 优先读取环境变量 MODEL_NAME，没有就用默认模型名
    model=os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V4-Flash"),
    # API Key 从环境变量读取；若没设置，os.environ[...] 会直接报 KeyError
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
)

# 创建 Agent。关键点是 tools=[...]：
# Agent 看到这两个函数的「名称 + docstring 参数说明」，
# 就能自己判断何时调用 calculate、何时调用 convert_currency。
agent = create_deep_agent(
    model=model,
    tools=[calculate, convert_currency],
    system_prompt="你是一个计算助手，能帮用户做数学运算和货币换算。",
)

# 让 Agent 跑一个需要「两步」的任务：先换汇，再乘系数。
# 它内部很可能会先调 convert_currency，再把结果交给 calculate。
# invoke 返回的是一个状态字典，其中 "messages" 是完整对话列表。
result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我把 100 美元换算成人民币，再用它乘以 1.08 的通胀系数。"}]}
)

# 列表最后一条消息就是模型的最终回答，.content 是它的文本内容
print(result["messages"][-1].content)
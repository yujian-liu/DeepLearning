from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print(f"add called: {a} + {b}")
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    print(f"multiply called: {a} + {b}")
    return a * b

if __name__ == "__main__":
    print("math")
    mcp.run(transport="stdio")
from enum import Enum
from pathlib import Path

import git
from git import Repo
from mcp.server import FastMCP
from pydantic import BaseModel

mcp = FastMCP('git_server')

# class GitInit(BaseModel):
#     repo_path: str
#
# class GitADD(BaseModel):
#     repo_path: str
#     files: list[str]
#
# class GitCommit(BaseModel):
#     repo_path: str
#     message: str
#
# class GitPush(BaseModel):
#     repo_path: str
#     branch: str
#
# class GitShowBranch(BaseModel):
#     repo_path: str
#
# class GitTools(str, Enum):
#     INIT = 'git_init'
#     ADD = 'git_add'
#     COMMIT = 'git_commit'
#     PUSH = 'git_push'
#     SHOW_BRANCH = 'git_show_branch'

@mcp.tool()
def git_init(repo_path: str) -> str:
    repo = Repo.init(Path(repo_path))
    if (Path(repo_path) / ".git").exists():
        return '创建成功'
    else:
        return '创建失败'

@mcp.tool()
def git_add(repo_path: str, files: list[str]) -> str:
    repo = git.Repo(repo_path)
    if files == ["."]:
        repo.git.add(".")
    else:
        repo.index.add(files)
    return "添加成功"

@mcp.tool()
def git_commit(repo_path: str, message: str) -> str:
    repo = git.Repo(repo_path)
    commit = repo.index.commit(message)
    return f"提交成功，hash={commit.hexsha}"

@mcp.tool()
def git_push(repo_path: str, branch: str) -> str:
    repo = git.Repo(repo_path)
    remote = repo.remote(name='origin')
    result = remote.push(refspec=branch)

    # 检查推送结果（是否成功）
    if result[0].flags & result[0].UP_TO_DATE:
        return "本地分支与远程已同步（无需推送）"
    elif result[0].flags & result[0].ERROR:
        return "推送失败！"
    else:
        return "推送成功！"

@mcp.tool()
def git_show_branch(repo_path: str) -> str:
    repo = git.Repo(repo_path)
    local_branches = repo.branches
    return ''.join([f'- {branch.name} \n' for branch in local_branches])

if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run(transport='stdio')

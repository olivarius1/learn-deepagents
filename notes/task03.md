# VFS
## what
虚拟文件系统, 通过提供一系列的文件操作接口,使agent和数据存储解耦
必要的api: read_file(path, offset, limit), write_file, edit_file, glob, grep, ls



## why
- agent上下文是有限的,每次会话都会带着之前的会话请求model api,  在长时间的agent活动中,上下文注定会用完,所以需要上下文"卸载" 和"压缩" 所以需要一个存储, 便于索引,按需按类存取. 

vfs带来的好处:
- 统一的工具接口, 屏蔽底层存储差异
- 安全与隔离, 避免直接暴露真实环境:如VfsBackend , sandbox
- 复杂agent能力的需要:skills, memory,代码执行,上下文管理都需要有一个文件系统
- 效率:为增量检索(渐进式检索) 提供能力支撑 如先ls -> glob -> grep ->read_file 提高上下文窗口利用率
- 可插拔带来灵活性

## 常用Backend
用法
```
from deepagents.backends import StateBackend, FilesystemBackend


backend = StateBackend()
agent = create_deep_agent(model=model, backend=backend)

```

### StateBackend
单会话


```python
{
  "/notes/a.txt": {
      "content": "hello\nworld\n",   # 文本，或 base64 编码的二进制
      "encoding": "utf-8",
      "created_at": "2026-11-23T...",
      "modified_at": "2026-11-23T...",
  },
  "/notes/b.txt": {...},
}
```

### FileSystemBackend
```
backend = FilesystemBackend(
    root_dir="./data",
    virtual_mode=False
)
```

### LocalShellBackend
相对 FileSystemBackend增加了execute, coding agent常用

### StoreBackend
基于langgraph Store, 跨会话, 基于DB
有namespace, 数据隔离更好

### CompositeBackend

带路由, 多种介质组合使用


### 沙箱
依赖第三方服务
如 modal, Daytona


> 生产环境 一般使用StoreBackend/CompositeBackend

### UDF
实现相关协议即可

## how

LLM: read_file(path="/notes/a.txt", offset=1, limit=2)
  ↓
FilesystemMiddleware
  ├─ validate_path()              路径合法性        ← 中间件
  ├─ _check_fs_permission()       权限              ← 中间件
  ├─ backend.read(...)            切出第 2~3 行      ← backend
  ├─ 空内容分支 / 视频分支 / base64 分支             ← 中间件
  ├─ format_content_with_line_numbers()  加 "  2→"  ← 中间件
  ├─ _truncate_paginated_read()   超 token 截断     ← 中间件
  └─ + "（还有 7 行未读，可继续 offset=3）"         ← 中间件
  ↓
LLM 看到带行号、带续读提示、尺寸可控的文本

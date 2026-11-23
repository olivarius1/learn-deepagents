from deepagents import create_deep_agent
from deepagents.backends import StateBackend, FilesystemBackend

model = ''
agent = create_deep_agent(model=model, backend=StateBackend())

backend = FilesystemBackend(
    root_dir="./data",
    virtual_mode=False
)
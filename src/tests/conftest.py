import datetime
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("CEREBRAS_API_KEY","test-key")

def _install_fake_module(name: str,**attrs)-> types.ModuleType:
    if name is sys.modules:
        return sys.modules[name]

    mod  = types.ModuleType(name)
    for k,v in attrs.items():
        setattr(mod,k,v)
    sys.modules[name] = mod
    return mod

_install_fake_module(
    "db.db",
    get_conn=MagicMock(name="get_conn"),
    put_conn=MagicMock(name="put_conn"),
    init_db=MagicMock(name="init_db"),
)

class _FakeTextEmbedding:
    def __init__(self,*a,**kw):
        pass

    def embed(self,texts,batch_size=8):
        # 384-dim equals to all-MiniLM-L6-v2
        for _ in texts:
            yield [0.0]*384


_install_fake_module("fastembed",TextEmbedding=_FakeTextEmbedding)

_fake_langfuse_client = MagicMock(name="langfuse_client")
_install_fake_module("langfuse",get_client=lambda:_fake_langfuse_client)

class FakePaper:

    def __init__(self,short_id, title,summary="summary text",authors=("A. Author",)):
        self._short_id = short_id
        self.title = title
        self.summary = summary
        self.authors=[types.SimpleNamespace(name=a) for a in authors]
        self.pdf_url=f"https://arxiv.org/pdf/{short_id}"
        self.published=datetime.datetime(2026,1,1)


    def get_short_id(self):
        return self._short_id


@pytest.fixture
def fake_paper_factory():
    return FakePaper
        

    
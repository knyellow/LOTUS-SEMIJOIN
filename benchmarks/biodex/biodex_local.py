import sys

from biodex_tester import BiodexTester, Join, JoinAndRerank

import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS


class LocalBiodexTester(BiodexTester):
    """BiodexTester with a local llama-server LLM and a local embedding model instead of OpenAI."""

    def set_configs(self):
        # keep the embedding model on CPU so the whole GPU is left for the LLM
        rm = SentenceTransformersRM(model="intfloat/e5-base-v2", max_batch_size=64, device="cpu")

        lm = LM(
            model="hosted_vllm/llama3.1-8b",  # llama.cpp llama-server, Q4_K_M
            api_base="http://localhost:8080/v1",
            max_batch_size=4,  # llama-server runs with a single slot
            temperature=0.0,
            max_tokens=256,
        )

        lotus.settings.configure(lm=lm, rm=rm, vs=FaissVS())


if __name__ == "__main__":
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    ts = LocalBiodexTester(n_samples=n_samples, truncation_limit=8000)

    name = "llama3.1-8b-q4"
    ts.add_pipeline(Join(recall_target=0.9, precision_target=0.9, name=name))
    ts.add_pipeline(JoinAndRerank(name=name))

    ts.test_pipelines()
    print(ts.summarize_pipeline_results("biodex_results", ts.n_samples, ["Join", "JoinAndRerank"]))

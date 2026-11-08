from dataclasses import replace


class Reranker:
    def __init__(self, docs, config):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.docs, self.config = torch, {d.doc_id: d for d in docs}, config
        kwargs = {'revision': config['revision'], 'trust_remote_code': False}
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'], **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model'], **kwargs)
        self.model.to(config['device']).eval()
        if config['fp16'] and config['device'].startswith('cuda'):
            self.model.half()
        self.resolved_revision = getattr(self.model.config, '_commit_hash', None)

    def rerank(self, query, candidates):
        depth = self.config.get('depth', len(candidates))
        if type(depth) is not int or depth < 1:
            raise ValueError('重排深度必须是正整数。')
        selected, tail = candidates[:depth], candidates[depth:]
        scored = []
        batch = self.config['batch_size']
        for start in range(0, len(selected), batch):
            hits = selected[start:start + batch]
            pairs = [(query, self.docs[h.doc_id].search_text) for h in hits]
            tokens = self.tokenizer(pairs, padding=True, truncation='only_second',
                                    max_length=self.config['max_length'], return_tensors='pt').to(self.config['device'])
            with self.torch.inference_mode():
                scores = self.model(**tokens).logits.reshape(-1).float().cpu().tolist()
            scored.extend(replace(h, score=float(s), rerank_score=float(s)) for h, s in zip(hits, scores))
        scored.sort(key=lambda h: (-h.score, h.doc_id))
        return [replace(hit, rank=rank) for rank, hit in enumerate(scored + tail, 1)]

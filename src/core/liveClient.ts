import type { Model } from './model.js';
export interface LiveStatus {
    revision: number;
    ready: boolean;
    loading: boolean;
    error: string | null;
}
/** Read-only loopback bridge. Only the user's Python process executes model code. */
export class LiveConnection {
    private running = false;
    private timer?: ReturnType<typeof setTimeout>;
    private controller?: AbortController;
    private revision = -1;
    private generation = 0;
    private base: string;
    constructor(url: string, private callbacks: {
        onModel: (model: Model) => void;
        onStatus?: (status: LiveStatus) => void;
        onError?: (error: Error) => void;
    }) {
        const parsed = new URL(url, location.href);
        if (!['127.0.0.1', 'localhost', '[::1]'].includes(parsed.hostname))
            throw Error('仅允许连接本机的模型服务。');
        this.base = parsed.origin;
    }
    start() { this.stop(); this.running = true; this.revision = -1; void this.poll(this.generation); return this; }
    private async poll(generation: number) {
        if (!this.running || generation !== this.generation)
            return;
        this.controller = new AbortController();
        try {
            const response = await fetch(this.base + '/api/status', { signal: this.controller.signal, cache: 'no-store' });
            if (!response.ok)
                throw Error(`本地服务返回 ${response.status}`);
            const status = await response.json() as LiveStatus;
            if (generation !== this.generation)
                return;
            this.callbacks.onStatus?.(status);
            if (status.ready && status.revision !== this.revision) {
                const r = await fetch(this.base + '/api/model', { signal: this.controller.signal, cache: 'no-store' });
                if (!r.ok)
                    throw Error('模型读取失败');
                const snapshot = await r.json();
                if (this.running && generation === this.generation) {
                    this.callbacks.onModel(snapshot.model);
                    this.revision = snapshot.revision;
                }
            }
        }
        catch (e) {
            if (this.running && generation === this.generation)
                this.callbacks.onError?.(e instanceof Error ? e : Error(String(e)));
        }
        finally {
            if (this.running && generation === this.generation)
                this.timer = setTimeout(() => void this.poll(generation), 800);
        }
    }
    async tensorWindow(id: string, indices: number[], signal?: AbortSignal): Promise<(number | null)[]> {
        const revision = this.revision, q = new URLSearchParams({ id, indices: indices.join(','), revision: String(revision) });
        const r = await fetch(this.base + '/api/tensor?' + q, { signal, cache: 'no-store' });
        const result = await r.json();
        if (!r.ok)
            throw Error(result.error ?? '张量窗口读取失败');
        if (result.revision !== this.revision || revision !== this.revision)
            throw Error('模型已更新，请重新读取窗口。');
        return result.values;
    }
    stop() {
        this.running = false;
        this.generation++;
        if (this.timer)
            clearTimeout(this.timer);
        this.controller?.abort();
    }
    get currentRevision() { return this.revision; }
}

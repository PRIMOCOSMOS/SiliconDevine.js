import type { Model } from './model.js';
export interface LiveStatus {
    revision: number;
    ready: boolean;
    loading: boolean;
    error: string | null;
    issue?: ImportIssue | null;
    importOptions?: ImportSchema | null;
    controlToken?: string | null;
}
export interface ImportField {
    name: string; kind: string; required?: boolean; enabled?: boolean; value?: unknown;
    annotation?: string; class?: string; choices?: unknown[]; children?: ImportField[]; parameterKind?: string;
}
export interface ImportSchema {
    entry?: string; label?: string; settings: Record<string, unknown>;
    candidates: {id:string;name:string;file:string;line:number}[];
    constructor: ImportField[]; inputs: ImportField[]; hasForward?: boolean;
}
export interface ImportIssue {
    code: string; title: string; message: string; detail?: string; file?: string; line?: number;
    module?: string; package?: string; requirements?: string[]; schema?: ImportSchema | null;
    fields?: string[];
}
/** Read-only loopback bridge. Only the user's Python process executes model code. */
export class LiveConnection {
    private running = false;
    private timer?: ReturnType<typeof setTimeout>;
    private controller?: AbortController;
    private revision = -1;
    private generation = 0;
    private base: string;
    private token?: string | null;
    private requests = new Set<AbortController>();
    constructor(url: string, private callbacks: {
        onModel: (model: Model) => void;
        onStatus?: (status: LiveStatus) => void;
        onError?: (error: Error) => void;
    }, private options: {requestTimeoutMs?: number; pollIntervalMs?: number} = {}) {
        const parsed = new URL(url, location.href);
        if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password || !['127.0.0.1', 'localhost', '[::1]'].includes(parsed.hostname))
            throw Error('仅允许连接本机的模型服务。');
        for(const value of Object.values(options))if(value!==undefined&&(!Number.isFinite(value)||value<=0))throw Error('连接超时和轮询间隔必须大于零。');
        this.base = parsed.origin;
    }
    private async request(path: string, init: RequestInit = {}): Promise<any> {
        const controller=new AbortController(),signal=init.signal;
        const abort=()=>controller.abort(signal?.reason);
        if(signal?.aborted)abort();else signal?.addEventListener('abort',abort,{once:true});
        const timeout=setTimeout(()=>controller.abort(new Error('本地服务响应超时，请检查 Python 进程后重试。')),this.options.requestTimeoutMs??15000);
        this.requests.add(controller);
        try{
            const response=await fetch(this.base+path,{...init,signal:controller.signal,cache:'no-store'});
            let result;
            try{result=await response.json();}catch{throw Error(`本地服务返回了无法读取的响应（HTTP ${response.status}）。`);}
            if(!response.ok)throw Error(typeof result?.error==='string'?result.error:`本地服务返回 ${response.status}`);
            return result;
        }finally{clearTimeout(timeout);signal?.removeEventListener('abort',abort);this.requests.delete(controller);}
    }
    start() { this.stop(); this.running = true; this.revision = -1; void this.poll(this.generation); return this; }
    private async poll(generation: number) {
        if (!this.running || generation !== this.generation)
            return;
        this.controller = new AbortController();
        try {
            const status = await this.request('/api/status', { signal: this.controller.signal }) as LiveStatus;
            if(!status||!Number.isSafeInteger(status.revision)||typeof status.ready!=='boolean')throw Error('本地服务状态格式无效。');
            if (generation !== this.generation)
                return;
            this.token=status.controlToken;
            this.callbacks.onStatus?.(status);
            if (status.ready && status.revision !== this.revision) {
                const snapshot = await this.request('/api/model', { signal: this.controller.signal });
                if(!snapshot||!Number.isSafeInteger(snapshot.revision)||!snapshot.model)throw Error('模型响应格式无效。');
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
                this.timer = setTimeout(() => void this.poll(generation), this.options.pollIntervalMs??800);
        }
    }
    async tensorWindow(id: string, indices: number[], signal?: AbortSignal): Promise<(number | null)[]> {
        const revision = this.revision, q = new URLSearchParams({ id, indices: indices.join(','), revision: String(revision) });
        const result = await this.request('/api/tensor?' + q, { signal });
        if (result.revision !== this.revision || revision !== this.revision)
            throw Error('模型已更新，请重新读取窗口。');
        if(!Array.isArray(result.values)||result.values.length!==indices.length)throw Error('张量窗口响应不完整。');
        return result.values;
    }
    async configure(settings?: Record<string, unknown>): Promise<void> {
        if(!this.token) throw Error('请在启动器打开的本地工作台中修改导入设置。');
        await this.request(settings?'/api/import-settings':'/api/retry',{
            method:'POST',headers:{'Content-Type':'application/json','X-SiliconDevine-Token':this.token},body:JSON.stringify(settings??{})
        });
    }
    stop() {
        this.running = false;
        this.generation++;
        if (this.timer)
            clearTimeout(this.timer);
        this.controller?.abort();
        for(const request of this.requests)request.abort();
        this.requests.clear();
        this.token=undefined;
    }
    get currentRevision() { return this.revision; }
}

"use client";

import { useRef, useState } from "react";
import type { ChangeEvent, DragEvent, FormEvent } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
const MAX_FILES = 5;
const MAX_FILE_SIZE = 5 * 1024 * 1024;
const EXAMPLE_QUESTIONS = [
  "What are the main points in these documents?",
  "What dates or deadlines should I know about?",
  "Summarize the recommendations with citations.",
];

type Source = { citation: number; id: string; source: string; page: number; snippet: string; score: number };
type Message = { id: string; role: "user" | "assistant"; text: string; sources?: Source[] };
type FileDetail = { name: string; pages: number };

const CODE_SEGMENT_RE = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g;

function linkCitationMarkers(markdown: string, sources: Source[] = []) {
  const availableCitations = new Set(sources.map((source) => source.citation));
  return markdown
    .split(CODE_SEGMENT_RE)
    .map((segment, index) => {
      if (index % 2 === 1) return segment;
      return segment.replace(/\u3010(\d+)\u3011|\[(\d+)\]/g, (marker, fullWidthNumber, plainNumber) => {
        const number = Number(fullWidthNumber || plainNumber);
        return availableCitations.has(number) ? `[${number}](#citation-${number})` : marker;
      });
    })
    .join("");
}

function friendlyError(status: number, detail?: string) {
  if (status === 413) return detail || "That PDF is too large. Each file must be 5 MB or smaller, with no more than 100 pages total.";
  if (status === 415) return detail || "Please choose PDF files only.";
  if (status === 429) return "You’ve sent several requests in a short time. Please wait a minute and try again.";
  if (status === 502) return detail || "The document index could not be built. Check that the embedding provider is available and try again.";
  if (status >= 500) return "The document service had a problem. Please try again in a moment.";
  if (status === 404 && detail === "session_expired") return "Your session expired (the server restarted). Please upload your documents again.";
  if (status === 404) return detail || "The requested document session was not found.";
  return detail || "Something went wrong. Please try again.";
}

async function responseError(response: Response) {
  let detail = "";
  try {
    const body = await response.json();
    detail = typeof body.detail === "string" ? body.detail : "";
  } catch {
    // Use the friendly status message when the server did not return JSON.
  }
  return friendlyError(response.status, detail);
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [fileDetails, setFileDetails] = useState<FileDetail[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [usingSample, setUsingSample] = useState(false);
  const [sessionExpired, setSessionExpired] = useState(false);
  const [activeSource, setActiveSource] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  function addFiles(incoming: FileList | File[]) {
    const selection = Array.from(incoming);
    setError("");
    if (selection.some((file) => file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf"))) {
      setError("Please choose PDF files only.");
      return;
    }
    if (selection.some((file) => !file.name.toLowerCase().endsWith(".pdf"))) {
      setError("Please choose PDF files only.");
      return;
    }
    if (selection.some((file) => file.size > MAX_FILE_SIZE)) {
      setError("One of those files is larger than 5 MB. Choose a smaller PDF.");
      return;
    }
    const combined = [...files, ...selection];
    if (combined.length > MAX_FILES) {
      setError("You can upload up to 5 PDFs at a time.");
      return;
    }
    setFiles(combined);
    setSessionId("");
    setFileDetails([]);
    setUsingSample(false);
    setSessionExpired(false);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) addFiles(event.target.files);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    setDragging(false);
    addFiles(event.dataTransfer.files);
  }

  async function indexDocuments() {
    if (!files.length || busy) return;
    setBusy(true);
    setError("");
    setMessages([]);
    setSessionId("");
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    try {
      const response = await fetch(`${API_URL}/upload`, { method: "POST", body: form });
      if (!response.ok) throw new Error(await responseError(response));
      const data = await response.json();
      setSessionId(data.session_id);
      setFileDetails(data.file_details || files.map((file: File) => ({ name: file.name, pages: 0 })));
      setUsingSample(false);
      setSessionExpired(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The server could not be reached. Check that the backend is running.");
    } finally {
      setBusy(false);
    }
  }

  async function useSampleDocuments() {
    if (busy) return;
    setBusy(true);
    setError("");
    setMessages([]);
    try {
      const response = await fetch(`${API_URL}/sample-session`);
      if (!response.ok) throw new Error(await responseError(response));
      const data = await response.json();
      setSessionId(data.session_id);
      setFileDetails(data.file_details || []);
      setFiles([]);
      setUsingSample(true);
      setSessionExpired(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The sample documents could not be loaded.");
    } finally {
      setBusy(false);
    }
  }

  async function sendQuestion(event?: FormEvent, preset?: string) {
    event?.preventDefault();
    const text = (preset ?? question).trim();
    if (!sessionId || !text || busy || text.length > 500) return;
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", text };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setBusy(true);
    setError("");
    setActiveSource("");
    try {
      let response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question: text }),
      });
      if (response.status === 404) {
        const body = await response.clone().json().catch(() => ({}));
        if (body.detail === "session_expired" && usingSample) {
          const sampleResponse = await fetch(`${API_URL}/sample-session`);
          if (!sampleResponse.ok) throw new Error(await responseError(sampleResponse));
          const sample = await sampleResponse.json();
          setSessionId(sample.session_id);
          setFileDetails(sample.file_details || []);
          response = await fetch(`${API_URL}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sample.session_id, question: text }),
          });
        } else if (body.detail === "session_expired") {
          setSessionId("");
          setSessionExpired(true);
          throw new Error(friendlyError(404, body.detail));
        }
      }
      if (!response.ok) throw new Error(await responseError(response));
      const data = await response.json();
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: "assistant", text: data.answer, sources: data.sources || [] },
      ]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The server could not be reached. Check that the backend is running.");
    } finally {
      setBusy(false);
    }
  }

  function clearChat() {
    setMessages([]);
    setError("");
    setActiveSource("");
  }

  function jumpToSource(messageId: string, citation: number) {
    const sourceId = `${messageId}-source-${citation}`;
    setActiveSource(sourceId);
    document.getElementById(sourceId)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function renderAnswer(message: Message) {
    const components: Components = {
      a: ({ href, children }) => {
        const citation = /^#citation-(\d+)$/.exec(href || "");
        if (citation) {
          const number = Number(citation[1]);
          const source = message.sources?.find((item) => item.citation === number);
          if (!source) return <>{children}</>;
          return (
            <button
              type="button"
              onClick={() => jumpToSource(message.id, number)}
              className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-sm bg-[#e4ebf3] px-1 align-baseline text-xs font-semibold text-[#173d66] hover:bg-[#d3dfec] focus:outline-none focus:ring-2 focus:ring-[#315f8b]"
              aria-label={`Jump to source ${number}`}
            >
              {children}
            </button>
          );
        }
        return <a href={href} className="text-[#173d66] underline underline-offset-2">{children}</a>;
      },
      table: ({ node, ...props }) => {
        void node;
        return (
          <div className="my-3 max-w-full overflow-x-auto">
            <table {...props} className="w-full border-collapse border border-[#d8d5ce] text-left text-xs leading-5" />
          </div>
        );
      },
      th: ({ node, ...props }) => {
        void node;
        return <th {...props} className="border border-[#d8d5ce] bg-[#f2f0ea] px-2 py-1.5 text-xs font-semibold" />;
      },
      td: ({ node, ...props }) => {
        void node;
        return <td {...props} className="border border-[#e2dfd8] px-2 py-1.5 align-top text-xs" />;
      },
      ul: ({ node, ...props }) => {
        void node;
        return <ul {...props} className="my-2 list-disc space-y-1 pl-6" />;
      },
      ol: ({ node, ...props }) => {
        void node;
        return <ol {...props} className="my-2 list-decimal space-y-1 pl-6" />;
      },
      p: ({ node, ...props }) => {
        void node;
        return <p {...props} className="my-2" />;
      },
      pre: ({ node, ...props }) => {
        void node;
        return <pre {...props} className="my-3 overflow-x-auto border border-[#e2dfd8] bg-[#f8f7f3] p-3 text-xs leading-5" />;
      },
    };

    return (
      <div className="text-sm leading-6 [&_strong]:font-semibold">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {linkCitationMarkers(message.text, message.sources)}
        </ReactMarkdown>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-[#f6f4ef] text-[#262b31]">
      <div className="mx-auto flex min-h-screen max-w-[1440px] flex-col border-x border-[#dedbd3] lg:flex-row">
        <aside className="w-full shrink-0 border-b border-[#dedbd3] bg-[#f2f0ea] p-5 lg:w-[320px] lg:border-b-0 lg:border-r lg:p-6">
          <div className="mb-8">
            <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#73777c]">Document Q&amp;A</p>
            <h1 className="mt-2 text-xl font-semibold tracking-[-0.03em]">Your documents</h1>
            <p className="mt-1 text-sm text-[#6c7177]">Upload PDFs to ask questions with citations.</p>
          </div>

          <div className="space-y-5">
            <div>
              <button
                type="button"
                onClick={() => fileInput.current?.click()}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                className={`flex min-h-32 w-full flex-col items-center justify-center border border-dashed px-4 py-5 text-center transition-colors ${dragging ? "border-[#173d66] bg-[#e8edf2]" : "border-[#c9c6bf] bg-[#f8f7f3] hover:border-[#173d66]"}`}
              >
                <span className="mb-2 flex h-8 w-8 items-center justify-center border border-[#d4d1ca] text-lg text-[#173d66]">＋</span>
                <span className="text-sm font-medium">Drop PDFs here or browse</span>
                <span className="mt-1 text-xs text-[#777c81]">Up to 5 files · 5 MB each</span>
              </button>
              <input ref={fileInput} type="file" accept="application/pdf,.pdf" multiple className="hidden" onChange={onFileChange} />
            </div>

            <section aria-label="Uploaded files">
              <div className="mb-2 flex items-center justify-between">
                <h2 className="text-xs font-semibold uppercase tracking-[0.1em] text-[#676d73]">Files</h2>
                {files.length > 0 && !sessionId && (
                  <button type="button" onClick={() => setFiles([])} className="text-xs text-[#68717a] underline underline-offset-2 hover:text-[#173d66]">Remove all</button>
                )}
              </div>
              {fileDetails.length > 0 ? (
                <ul className="divide-y divide-[#dedbd3] border-y border-[#dedbd3]">
                  {fileDetails.map((file, index) => (
                    <li key={`${file.name}-${index}`} className="flex items-start justify-between gap-3 py-2.5">
                      <span className="min-w-0 break-all font-mono text-xs leading-5 text-[#333941]">{file.name}</span>
                      <span className="shrink-0 pt-0.5 text-xs text-[#777c81]">{file.pages} {file.pages === 1 ? "page" : "pages"}</span>
                    </li>
                  ))}
                </ul>
              ) : files.length ? (
                <ul className="divide-y divide-[#dedbd3] border-y border-[#dedbd3]">
                  {files.map((file, index) => (
                    <li key={`${file.name}-${index}`} className="flex items-start justify-between gap-3 py-2.5">
                      <span className="min-w-0 break-all font-mono text-xs leading-5">{file.name}</span>
                      <span className="shrink-0 text-xs text-[#777c81]">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="border-y border-[#dedbd3] py-3 text-xs text-[#777c81]">No files selected.</p>
              )}
            </section>

            <button
              type="button"
              disabled={!files.length || busy}
              onClick={indexDocuments}
              className="w-full border border-[#173d66] bg-[#173d66] px-4 py-2.5 text-sm font-medium text-white hover:bg-[#214d79] disabled:cursor-not-allowed disabled:border-[#c5c7c9] disabled:bg-[#c5c7c9]"
            >
              {busy && files.length ? "Indexing…" : "Index documents"}
            </button>

            <button
              type="button"
              onClick={useSampleDocuments}
              disabled={busy}
              className="w-full text-left text-sm text-[#173d66] underline decoration-[#9aabba] underline-offset-4 hover:decoration-[#173d66] disabled:opacity-50"
            >
              Use sample documents →
            </button>

            {sessionId && (
              <div className="border-t border-[#dedbd3] pt-4">
                <p className="text-xs font-medium text-[#4d5964]">{usingSample ? "Sample documents ready" : "Documents indexed"}</p>
                <p className="mt-1 font-mono text-[10px] text-[#85898c]">{sessionId}</p>
              </div>
            )}
          </div>
        </aside>

        <section className="flex min-h-[70vh] min-w-0 flex-1 flex-col lg:min-h-screen" aria-label="Document chat">
          <header className="flex h-[65px] shrink-0 items-center justify-between border-b border-[#dedbd3] px-5 sm:px-8">
            <div>
              <h2 className="text-sm font-semibold">Ask your documents</h2>
              <p className="mt-0.5 text-xs text-[#7a7e82]">Answers include page level sources</p>
            </div>
            <button type="button" onClick={clearChat} disabled={!messages.length && !error} className="border border-[#d7d4cd] px-3 py-1.5 text-xs text-[#626970] hover:border-[#9aa3aa] hover:text-[#252b31] disabled:cursor-not-allowed disabled:opacity-40">Clear chat</button>
          </header>

          <div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
            {messages.length === 0 ? (
              <div className="mx-auto flex min-h-[45vh] max-w-xl flex-col items-center justify-center text-center">
                <div className="mb-4 flex h-11 w-11 items-center justify-center border border-[#d6d3cc] font-mono text-sm text-[#173d66]">Q&amp;A</div>
                <h3 className="text-base font-semibold">A clear answer, with its source.</h3>
                <p className="mt-2 max-w-sm text-sm leading-6 text-[#74797e]">{sessionId ? "Ask a question about the indexed documents." : "Choose PDFs or use the sample documents to start a conversation."}</p>
                {sessionId && usingSample && (
                  <div className="mt-6 flex flex-wrap justify-center gap-2">
                    {EXAMPLE_QUESTIONS.map((sample) => (
                      <button key={sample} type="button" onClick={() => void sendQuestion(undefined, sample)} disabled={busy} className="border border-[#d8d5ce] bg-[#f9f8f4] px-3 py-2 text-xs text-[#47515a] hover:border-[#9baabb] hover:text-[#173d66] disabled:opacity-50">{sample}</button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="mx-auto max-w-3xl space-y-7">
                {messages.map((message) => (
                  <article key={message.id} className={message.role === "user" ? "ml-auto max-w-[85%]" : "max-w-[95%]"}>
                    <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.1em] text-[#85898c]">{message.role === "user" ? "You" : "Answer"}</p>
                    <div className={message.role === "user" ? "border border-[#dcd9d2] bg-[#eeece6] px-4 py-3 text-sm leading-6" : "text-[15px] leading-7 text-[#30363c]"}>
                      {message.role === "assistant" ? renderAnswer(message) : message.text}
                    </div>
                    {message.role === "assistant" && !!message.sources?.length && (
                      <section className="mt-5 border-t border-[#dedbd3] pt-3" aria-label="Sources">
                        <h4 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.1em] text-[#777d82]">Sources</h4>
                        <ol className="space-y-2">
                          {message.sources.map((source) => {
                            const sourceId = `${message.id}-source-${source.citation}`;
                            const active = activeSource === sourceId;
                            return (
                              <li id={sourceId} key={source.id} className={`border px-3 py-2.5 transition-colors ${active ? "border-[#7993ad] bg-[#eaf0f5]" : "border-[#e2dfd8] bg-[#f9f8f4]"}`}>
                                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-xs text-[#344b61]">
                                  <span className="font-semibold">[{source.citation}]</span>
                                  <span className="break-all">{source.source}</span>
                                  <span className="text-[#777d82]">· p. {source.page}</span>
                                </div>
                                <p className="mt-1.5 text-xs leading-5 text-[#6f7479]">{source.snippet}</p>
                              </li>
                            );
                          })}
                        </ol>
                      </section>
                    )}
                  </article>
                ))}
                {busy && (
                  <div className="flex items-center gap-2 text-sm text-[#7a7e82]" role="status" aria-live="polite">
                    <span className="flex gap-1"><i className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#173d66]" /><i className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#173d66] [animation-delay:150ms]" /><i className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#173d66] [animation-delay:300ms]" /></span>
                    Reading the documents…
                  </div>
                )}
              </div>
            )}
          </div>

          {error && <div role="alert" className="mx-5 mb-3 flex flex-wrap items-center justify-between gap-3 border border-[#d8aaa2] bg-[#fbf1ee] px-3 py-2.5 text-sm text-[#8b3f32] sm:mx-8"><span>{error}</span>{sessionExpired && <button type="button" disabled={!files.length || busy} onClick={() => void indexDocuments()} className="shrink-0 border border-[#8b3f32] px-3 py-1.5 text-xs font-medium hover:bg-[#f5e4df] disabled:opacity-50">Upload documents again</button>}</div>}

          <form onSubmit={(event) => void sendQuestion(event)} className="border-t border-[#dedbd3] px-5 py-4 sm:px-8">
            <div className="mx-auto flex max-w-3xl items-end gap-2 border border-[#d5d2cb] bg-[#fbfaf7] p-2 focus-within:border-[#7993ad]">
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value.slice(0, 500))}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void sendQuestion();
                  }
                }}
                disabled={!sessionId || busy}
                maxLength={500}
                rows={1}
                placeholder={sessionId ? "Ask a question about your documents…" : "Index documents to start asking questions"}
                className="max-h-32 min-h-10 flex-1 resize-y bg-transparent px-2 py-2 text-sm leading-5 outline-none placeholder:text-[#92969a] disabled:cursor-not-allowed"
                aria-label="Your question"
              />
              <span className="pb-3 text-[10px] tabular-nums text-[#999da0]">{question.length}/500</span>
              <button type="submit" disabled={!sessionId || !question.trim() || busy} className="h-9 shrink-0 border border-[#173d66] bg-[#173d66] px-3 text-sm font-medium text-white hover:bg-[#214d79] disabled:cursor-not-allowed disabled:border-[#c5c7c9] disabled:bg-[#c5c7c9]">Send</button>
            </div>
            <p className="mx-auto mt-2 max-w-3xl text-[10px] text-[#92969a]">Enter to send · Shift + Enter for a new line</p>
          </form>
        </section>
      </div>
    </main>
  );
}

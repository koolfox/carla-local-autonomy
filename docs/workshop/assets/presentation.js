import { isolateScientificText, scientificDiagramSource, setDiagramDirection } from './text-direction.js';
/** Audience slides and the research reader share one Markdown source. */
const source = new URL('../docs/class-presentation.fa.md', import.meta.url);
const project = new URL('../', import.meta.url);
const deck = document.getElementById('deck');
const status = document.getElementById('deck-status');
const select = document.getElementById('slide-select');
const previous = document.getElementById('slide-prev');
const next = document.getElementById('slide-next');
let slides = [];
let current = 0;
let diagramLibrary;
const fa = number => new Intl.NumberFormat('fa-IR').format(number);

function textElement(tag, text, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

function addSlideIdentity(slide, details) {
  const header = document.createElement('header');
  header.className = 'slide-header';
  const university = document.createElement('div');
  university.append(
    textElement('p', details.institution, 'slide-institution'),
    textElement('p', 'درس ' + details.course, 'slide-course'),
  );
  /* The workshop has no supplied institutional logo. */
  const logo = document.createElement('img');
  logo.className = 'university-logo';
  logo.src = new URL(details.logo.path, project).href;
  logo.alt = 'نشان ' + details.institution;
  logo.width = details.logo.width;
  logo.height = details.logo.height;
  header.append(university);
  const content = document.createElement('div');
  content.className = 'slide-content';
  content.append(...slide.childNodes);
  const footer = document.createElement('footer');
  footer.className = 'slide-footer';
  footer.append(textElement('p', details.short_title), textElement('p', details.season));
  slide.append(header, content, footer);
}

function createCover(details) {
  const cover = document.createElement('section');
  cover.className = 'slide cover';
  cover.hidden = true;
  cover.append(textElement('h1', details.title), textElement('p', details.subtitle, 'cover-subtitle'));
  const identity = document.createElement('div');
  identity.className = 'cover-details';
  const supervisor = document.createElement('p');
  const profile = document.createElement('a');
  profile.className = 'supervisor-profile';
  profile.href = details.supervisor_profile_url;
  profile.textContent = 'استاد راهنما: ' + details.supervisor;
  supervisor.append(profile);
  identity.append(
    textElement('p', details.program),
    textElement('p', 'ارائه‌دهنده: ' + details.presenter),
    supervisor,
  );
  cover.append(identity);
  return cover;
}

function loadDiagrams() {
  if (!diagramLibrary) diagramLibrary = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = new URL('vendor/mermaid/mermaid.min.js', import.meta.url);
    script.onload = () => {
      window.mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'neutral', fontFamily: 'arad', flowchart: { htmlLabels: false } });
      resolve(window.mermaid);
    };
    script.onerror = () => reject(new Error('diagram unavailable'));
    document.head.append(script);
  });
  return diagramLibrary;
}

async function renderDiagrams() {
  const diagrams = [...deck.querySelectorAll('pre code.language-mermaid')];
  if (!diagrams.length) return;
  try {
    const mermaid = await loadDiagrams();
    await document.fonts.ready;
    for (const [index, code] of diagrams.entries()) {
      const { svg } = await mermaid.render('slide-diagram-' + index, scientificDiagramSource(code.textContent));
      const figure = document.createElement('figure');
      figure.className = 'slide-diagram';
      figure.innerHTML = svg;
      const graphic = figure.querySelector('svg');
      graphic.setAttribute('role', 'img');
      graphic.setAttribute('aria-label', 'نمودار مفهومی روابط توضیح‌داده‌شده در این اسلاید؛ مقیاس اتمی ندارد');
      setDiagramDirection(graphic);
      code.parentElement.replaceWith(figure);
    }
  } catch {
    for (const code of deck.querySelectorAll('pre code.language-mermaid')) {
      const note = document.createElement('p');
      note.textContent = 'نمودار آماده نشد؛ رابطه‌های آن در متن همین اسلاید توضیح داده شده‌اند.';
      code.parentElement.replaceWith(note);
    }
    deck.dataset.diagram = 'fallback';
  }
}

const mediaDialog = document.getElementById('slide-media-dialog');
const mediaStage = document.getElementById('slide-media-stage');
let mediaTrigger;
let mediaHome;
let activeModel;

function closeMedia() { mediaDialog.close(); }
mediaDialog.addEventListener('close', () => {
  if (activeModel && mediaHome) mediaHome.append(activeModel);
  mediaStage.replaceChildren();
  activeModel = null;
  mediaTrigger?.focus();
});
document.getElementById('close-slide-media').addEventListener('click', closeMedia);

function openMedia(figure, trigger, model = false) {
  mediaTrigger = trigger;
  document.getElementById('slide-media-title').textContent = model ? 'مدل سه‌بعدی؛ مختصات تجربی' : 'نمای بزرگ شکل';
  document.getElementById('slide-media-caption').textContent = figure.querySelector('figcaption')?.textContent || 'نمودار مفهومی همین اسلاید';
  mediaStage.replaceChildren();
  if (model) {
    activeModel = figure.querySelector('iframe');
    mediaHome = figure;
    mediaStage.append(activeModel);
  } else {
    const graphic = figure.querySelector('img, svg').cloneNode(true);
    if (graphic.tagName.toLowerCase() === 'svg') {
      const ids = new Map([...graphic.querySelectorAll('[id]')].map(node => [node.id, node.id + '-enlarged']));
      if (graphic.id) { ids.set(graphic.id, graphic.id + '-enlarged'); graphic.id += '-enlarged'; }
      for (const node of graphic.querySelectorAll('*')) {
        if (node.id) node.id = ids.get(node.id);
        for (const attr of [...node.attributes]) {
          if (attr.name === 'id') continue;
          let value = attr.value;
          for (const [id, replacement] of ids) value = value.replaceAll('#' + id, '#' + replacement);
          if (value !== attr.value) node.setAttribute(attr.name, value);
        }
      }
    }
    mediaStage.append(graphic);
  }
  mediaDialog.showModal();
}

function addMediaControls(figure) {
  const controls = document.createElement('div');
  controls.className = 'slide-media-controls';
  const enlarge = textElement('button', 'بزرگ‌نمایی شکل'); enlarge.type = 'button';
  enlarge.addEventListener('click', () => openMedia(figure, enlarge));
  controls.append(enlarge);
  const image = figure.querySelector('img');
  const modelId = image?.src.endsWith('complex-preview.png') ? '3WD5'
    : image?.src.endsWith('structure-comparison.png') ? '4FM9' : null;
  if (modelId) {
    const model = textElement('button', 'مدل سه‌بعدی ' + modelId); model.type = 'button';
    model.addEventListener('click', () => {
      let frame = figure.querySelector('iframe');
      if (!frame) {
        frame = document.createElement('iframe');
        frame.title = 'مدل سه‌بعدی ' + modelId + '؛ چرخش با ماوس یا کلیدها';
        frame.src = new URL('assets/reference-viewer.html?pdb=' + modelId, project).href;
        figure.append(frame);
      }
      openMedia(figure, model, true);
    });
    controls.append(model);
  }
  figure.append(controls);
}

/** Slide order and chapter exports share the one course-map.json; week is a legacy alias. */
async function applyChapters() { slides.forEach((slide, i) => slide.id = 'slide-' + (i + 1)); }

function hashIndex() {
  const number = Number(location.hash.match(/^#slide-(\d+)$/)?.[1]);
  return Number.isInteger(number) && number >= 1 && number <= slides.length ? number - 1 : 0;
}

function show(index, focus = false) {
  if (!slides.length) return;
  current = Math.max(0, Math.min(slides.length - 1, index));
  for (const [number, slide] of slides.entries()) {
    if (number !== current) slide.querySelectorAll("video").forEach(video => video.pause());
    slide.hidden = number !== current;
  }
  select.value = String(current);
  previous.disabled = current === 0;
  next.disabled = current === slides.length - 1;
  document.getElementById('slide-progress').textContent = `${fa(current + 1)} / ${fa(slides.length)}`;

  history.replaceState(null, '', '#slide-' + (current + 1));
  document.title = slides[current].querySelector('h1').textContent + ' · کارگاه CARLA';
  if (focus) { deck.focus({ preventScroll: true }); window.scrollTo({ top: 0, behavior: 'instant' }); }
}

async function load() {
  deck.setAttribute('aria-busy', 'true');
  document.getElementById('retry-deck').hidden = true;
  delete status.dataset.weekError;
  status.textContent = 'در حال آماده‌سازی اسلایدها…';
  try {
    const [response, metadata, slotsResponse] = await Promise.all([
      fetch(source, { signal: AbortSignal.timeout(15000) }),
      fetch(new URL('data/metadata/presentation.json', project), { signal: AbortSignal.timeout(15000) }),
      fetch(new URL('data/metadata/media-slots.json', project), { signal: AbortSignal.timeout(15000) }),
    ]);
    if (!response.ok || !metadata.ok || !slotsResponse.ok) throw new Error('source unavailable');
    const slots = await slotsResponse.json();
    const details = await metadata.json();
    const markdown = window.markdownit({ html: false, linkify: false });
    const documentRoot = document.createElement('div');
    documentRoot.innerHTML = markdown.render(await response.text());
    deck.replaceChildren(createCover(details));
    let section;
    let body;
    let chapter = 0;
    for (const child of [...documentRoot.children]) {
      if (child.tagName === 'H2') {
        section = document.createElement('section'); section.className = 'slide'; section.hidden = true;
        section.dataset.chapter = String(++chapter);
        const title = document.createElement('h1'); title.textContent = child.textContent.replace(/^[\d۰-۹]+[.ـ\s]+/, '');
        body = document.createElement('div'); body.className = 'slide-body';
        section.append(title, body); deck.append(section);
      } else if (body) body.append(child);
    }
    slides = [...deck.children];
    if (!slides.length) throw new Error('no slides');
    // Local historical clips are decoded only on request; the caption is also a text alternative.
    for (const link of deck.querySelectorAll('a[href^="../assets/video/"][href$=".mp4"]')) {
      const figure = document.createElement('figure');
      figure.className = 'workshop-video';
      const video = document.createElement('video');
      video.controls = true; video.playsInline = true; video.preload = 'none';
      video.src = new URL(link.getAttribute('href'), source).href;
      video.poster = video.src.replace(/\.mp4$/, '.jpg');
      video.setAttribute('aria-label', link.textContent.trim());
      const caption = document.createElement('figcaption');
      caption.textContent = link.textContent.trim() + ' · نمونهٔ تاریخی، بی‌صدا؛ برای پخش کلیک کنید.';
      const fallback = document.createElement('a');
      fallback.href = video.src; fallback.textContent = 'دریافت فیلم';
      fallback.setAttribute('download', '');
      video.addEventListener('error', () => { caption.textContent = 'پخش ویدیو ممکن نیست؛ از پیوند دریافت فیلم استفاده کنید.'; });
      const provenance = document.createElement('a');
      provenance.href = 'https://github.com/koolfox/carla-local-autonomy/blob/main/samples/README.md';
      provenance.textContent = 'منبع و محدودیت‌ها'; provenance.target = '_blank'; provenance.rel = 'noopener noreferrer';
      figure.append(video, caption, fallback, provenance);
      link.parentElement.replaceWith(figure);
    }
    // A slot stays visibly empty until an individually selected local asset is supplied.
    for (const link of deck.querySelectorAll('a[href^="#slot-"]')) {
      const id = link.getAttribute('href').slice(6);
      const item = slots[id] || {};
      const figure = document.createElement('figure');
      figure.className = 'media-slot'; figure.dataset.slot = id;
      const caption = document.createElement('figcaption'); caption.textContent = link.textContent.trim();
      let asset = null;
      if (typeof item.src === 'string' && ['image', 'video'].includes(item.type)) {
        let url;
        try { url = new URL(item.src, project); } catch { /* Invalid paths remain empty slots. */ }
        if (url && url.origin === project.origin && url.pathname.startsWith(project.pathname + 'assets/')) {
          asset = document.createElement(item.type === 'video' ? 'video' : 'img');
          asset.src = url.href;
          if (item.type === 'video') { asset.controls = true; asset.playsInline = true; asset.preload = 'none'; asset.setAttribute('aria-label', item.alt || caption.textContent); }
          else asset.alt = item.alt || caption.textContent;
        }
      }
      const placeholder = (unavailable = false) => {
        const block = document.createElement('div'); block.className = 'media-placeholder';
        const heading = document.createElement('strong'); heading.textContent = 'جای رسانهٔ انتخابی';
        const hint = document.createElement('span');
        hint.textContent = unavailable ? 'فایل رسانه در دسترس نیست؛ مسیر و قالب آن را بررسی کنید.' : (item.purpose || caption.textContent);
        block.append(heading, hint); return block;
      };
      if (asset) {
        asset.addEventListener('error', () => asset.replaceWith(placeholder(true)), { once: true });
        figure.append(asset);
      } else figure.append(placeholder());
      figure.append(caption); link.parentElement.replaceWith(figure);
    }
    // The notebook keeps its lab link; slides offer the same model in place.
    for (const link of deck.querySelectorAll('a[href$="#lab"]')) link.parentElement.remove();
    for (const link of deck.querySelectorAll('a[href]')) {
      if (link.hasAttribute('download')) continue;
      const destination = new URL(link.getAttribute('href'), source);
      link.href = destination.href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
    }
    for (const img of deck.querySelectorAll('img')) {
      if (img.closest('.media-slot')) continue;
      img.src = new URL(img.getAttribute('src'), source).href;
      const figure = document.createElement('figure');
      const caption = document.createElement('figcaption'); caption.textContent = img.alt;
      const paragraph = img.parentElement;
      figure.append(img, caption);
      if (paragraph.tagName === 'P' && !paragraph.textContent.trim()) paragraph.replaceWith(figure);
      else paragraph.after(figure);
      img.addEventListener('error', () => { img.hidden = true; caption.textContent = 'شکل در دسترس نیست: ' + img.alt; });
    }
    await renderDiagrams();
    for (const slide of slides) {
      const body = slide.querySelector('.slide-body');
      const citations = [...(body?.querySelectorAll(':scope > blockquote') || [])];
      const references = document.createElement('aside');
      references.className = 'slide-references';
      references.setAttribute('aria-label', 'منابع و حدود شواهد این اسلاید');
      references.append(...citations);
      const figure = body?.querySelector(':scope > figure');
      if (figure) {
        slide.classList.add('has-media');
        const copy = document.createElement('div'); copy.className = 'slide-copy';
        for (const child of [...body.childNodes]) if (child !== figure) copy.append(child);
        body.replaceChildren(copy, figure);
      }
      if (citations.length) slide.append(references);
      for (const media of slide.querySelectorAll('figure:not(.workshop-video):not(.media-slot)')) addMediaControls(media);
      addSlideIdentity(slide, details);
      if (body) {

        for (const link of body.querySelectorAll('a.term-link')) {
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
        }
      }
      isolateScientificText(slide);
    }
    const sourcesSlide = slides.find((slide) => (slide.querySelector('h1')?.textContent || '').trim().startsWith('منابع')) || slides.at(-1);
    sourcesSlide.classList.add('sources');
    await applyChapters(details);
    select.replaceChildren(...slides.map((slide, index) => new Option(`${fa(index + 1)}. ${slide.querySelector('h1').textContent}`, String(index))));
    select.disabled = false;
    document.getElementById('print-slides').disabled = false;
    document.getElementById('sources-link').href = '#slide-' + (slides.indexOf(sourcesSlide) + 1);
    if (!status.dataset.weekError) status.textContent = '';
    deck.dataset.state = 'ready';
    show(hashIndex());
  } catch {
    status.textContent = 'اسلایدها بارگذاری نشدند. دوباره تلاش کنید یا متن ارائه را در کتابچه باز کنید.';
    deck.dataset.state = 'error';
    document.getElementById('retry-deck').hidden = false;
  } finally { deck.setAttribute('aria-busy', 'false'); }
}
previous.addEventListener('click', () => show(current - 1, true));
next.addEventListener('click', () => show(current + 1, true));
select.addEventListener('change', () => show(Number(select.value), true));
addEventListener('hashchange', () => show(hashIndex(), true));
addEventListener('keydown', event => {
  if (mediaDialog.open) return;
  if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest('select, input, textarea, video, [contenteditable]')) return;
  const changes = { ArrowLeft: 1, ArrowRight: -1, PageDown: 1, PageUp: -1 };
  if (event.key in changes) { event.preventDefault(); show(current + changes[event.key], true); }
  if (event.key === 'Home' || event.key === 'End') { event.preventDefault(); show(event.key === 'Home' ? 0 : slides.length - 1, true); }
});
document.getElementById('fullscreen').hidden = !document.fullscreenEnabled;
document.getElementById('fullscreen').addEventListener('click', async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch { status.textContent = 'مرورگر اجازهٔ تمام‌صفحه نداد؛ نمایش معمولی همچنان در دسترس است.'; }
});
addEventListener('fullscreenchange', () => { document.getElementById('fullscreen').textContent = document.fullscreenElement ? 'خروج از تمام‌صفحه' : 'تمام‌صفحه'; });
document.getElementById('print-slides').addEventListener('click', () => window.print());
document.getElementById('retry-deck').addEventListener('click', load);
load();

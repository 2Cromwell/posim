/* ============================================================
   POSIM Academic Project Page — JavaScript
   ============================================================ */

// ---------- i18n (Language Switching) ----------
let currentLang = 'en';

function setLang(lang) {
  currentLang = lang;
  document.documentElement.lang = lang === 'cn' ? 'zh-CN' : 'en';

  document.getElementById('langEN').classList.toggle('active', lang === 'en');
  document.getElementById('langCN').classList.toggle('active', lang === 'cn');

  document.querySelectorAll('[data-en][data-cn]').forEach(el => {
    const text = lang === 'en' ? el.getAttribute('data-en') : el.getAttribute('data-cn');
    if (text !== null) el.innerHTML = text;
  });

  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false }
      ],
      throwOnError: false
    });
  }
}

// ---------- Reading Progress Bar ----------
(function() {
  const bar = document.createElement('div');
  bar.className = 'reading-progress';
  bar.style.width = '0%';
  document.body.appendChild(bar);

  window.addEventListener('scroll', () => {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
    bar.style.width = Math.min(progress, 100) + '%';
  }, { passive: true });
})();

// ---------- Navbar Scroll Effect ----------
const navbar = document.getElementById('navbar');

window.addEventListener('scroll', () => {
  const scrollY = window.scrollY;
  navbar.classList.toggle('scrolled', scrollY > 10);
  document.getElementById('backToTop').classList.toggle('show', scrollY > 400);
  updateActiveNavLink();
}, { passive: true });

// ---------- Active Nav Link ----------
function updateActiveNavLink() {
  const sections = document.querySelectorAll('section[id]');
  const scrollPos = window.scrollY + 100;

  sections.forEach(section => {
    const top = section.offsetTop;
    const height = section.offsetHeight;
    const id = section.getAttribute('id');
    const link = document.querySelector(`.navbar-links a[href="#${id}"]`);

    if (link) {
      if (scrollPos >= top && scrollPos < top + height) {
        document.querySelectorAll('.navbar-links a').forEach(a => a.classList.remove('active'));
        link.classList.add('active');
      }
    }
  });
}

// ---------- Mobile Menu ----------
function toggleMobileMenu() {
  document.getElementById('navLinks').classList.toggle('open');
}
document.querySelectorAll('.navbar-links a').forEach(link => {
  link.addEventListener('click', () => {
    document.getElementById('navLinks').classList.remove('open');
  });
});

// ---------- Lightbox ----------
function openLightbox(imgEl) {
  if (!imgEl) return;
  const lightbox = document.getElementById('lightbox');
  const lightboxImg = document.getElementById('lightboxImg');
  lightboxImg.src = imgEl.src;
  lightboxImg.alt = imgEl.alt || 'Figure';
  lightbox.classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
  document.body.style.overflow = '';
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
});

// ---------- Copy BibTeX ----------
function copyBibtex() {
  const code = document.getElementById('bibtexCode').textContent;
  navigator.clipboard.writeText(code).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = currentLang === 'cn' ? '已复制 ✓' : 'Copied ✓';
    btn.style.background = 'rgba(34,197,94,0.3)';
    setTimeout(() => {
      btn.textContent = currentLang === 'cn' ? '复制' : 'Copy';
      btn.style.background = '';
    }, 2000);
  }).catch(() => {
    const textarea = document.createElement('textarea');
    textarea.value = code;
    textarea.style.cssText = 'position:fixed;opacity:0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    const btn = document.getElementById('copyBtn');
    btn.textContent = currentLang === 'cn' ? '已复制 ✓' : 'Copied ✓';
    setTimeout(() => { btn.textContent = currentLang === 'cn' ? '复制' : 'Copy'; }, 2000);
  });
}

// ---------- Scroll Reveal with stagger ----------
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.08, rootMargin: '0px 0px -30px 0px' });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

// ---------- Smooth scroll for anchor links ----------
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    const targetId = this.getAttribute('href');
    if (targetId === '#') return;
    e.preventDefault();
    const target = document.querySelector(targetId);
    if (target) {
      const offset = navbar.offsetHeight + 16;
      const top = target.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top, behavior: 'smooth' });
    }
  });
});

// ---------- Particle System (Hero Background) ----------
(function() {
  const canvas = document.getElementById('particleCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const hero = document.getElementById('hero');
  let particles = [];
  let animId;
  let w, h;

  function resize() {
    w = canvas.width = hero.offsetWidth;
    h = canvas.height = hero.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x = Math.random() * w;
      this.y = Math.random() * h;
      this.size = Math.random() * 2.5 + 0.5;
      this.speedX = (Math.random() - 0.5) * 0.4;
      this.speedY = (Math.random() - 0.5) * 0.4;
      this.opacity = Math.random() * 0.35 + 0.08;
    }
    update() {
      this.x += this.speedX;
      this.y += this.speedY;
      if (this.x < 0 || this.x > w || this.y < 0 || this.y > h) this.reset();
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(26, 86, 219, ${this.opacity})`;
      ctx.fill();
    }
  }

  const count = Math.min(Math.floor(w * h / 12000), 80);
  for (let i = 0; i < count; i++) particles.push(new Particle());

  function drawLines() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(26, 86, 219, ${0.06 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
  }

  function animate() {
    ctx.clearRect(0, 0, w, h);
    particles.forEach(p => { p.update(); p.draw(); });
    drawLines();
    animId = requestAnimationFrame(animate);
  }
  animate();

  // Pause when hero not visible
  const heroObs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { if (!animId) animate(); }
      else { cancelAnimationFrame(animId); animId = null; }
    });
  }, { threshold: 0.05 });
  heroObs.observe(hero);
})();

// ---------- Card Tilt Effect ----------
document.querySelectorAll('.contrib-card').forEach(card => {
  card.addEventListener('mousemove', e => {
    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const rotateX = ((y - centerY) / centerY) * -4;
    const rotateY = ((x - centerX) / centerX) * 4;
    card.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-2px)`;
  });
  card.addEventListener('mouseleave', () => {
    card.style.transform = '';
  });
});

// ---------- Pipeline step hover ----------
document.querySelectorAll('.pipeline-step').forEach(step => {
  step.addEventListener('mouseenter', () => {
    step.style.transform = 'translateY(-5px) scale(1.08)';
  });
  step.addEventListener('mouseleave', () => {
    step.style.transform = '';
  });
});

// ---------- Animated counter for metric highlights ----------
function animateCounters() {
  document.querySelectorAll('.metric-highlight-card .value').forEach(el => {
    const text = el.textContent;
    if (el.dataset.animated) return;

    const match = text.match(/\+?(\d+\.?\d*)/);
    if (!match) return;

    const target = parseFloat(match[1]);
    const prefix = text.startsWith('+') ? '+' : '';
    const suffix = text.replace(/\+?\d+\.?\d*/, '');
    const duration = 1500;
    const startTime = performance.now();
    el.dataset.animated = 'true';
    el.classList.add('counting');

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = (target * eased).toFixed(1);
      el.textContent = prefix + current + suffix;
      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        el.classList.remove('counting');
      }
    }
    requestAnimationFrame(update);
  });
}

const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateCounters();
      counterObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.3 });
const metricSection = document.querySelector('.metric-highlights');
if (metricSection) counterObserver.observe(metricSection);

// ---------- Dataset stat counter animation ----------
const datasetStatObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.querySelectorAll('.dataset-stat strong').forEach(el => {
        if (el.dataset.animated) return;
        const text = el.textContent;
        const numMatch = text.match(/([\d,]+)/);
        if (!numMatch) return;
        const target = parseInt(numMatch[1].replace(/,/g, ''));
        const prefix = text.substring(0, text.indexOf(numMatch[1]));
        const suffix = text.substring(text.indexOf(numMatch[1]) + numMatch[1].length);
        const duration = 1200;
        const startTime = performance.now();
        el.dataset.animated = 'true';

        function update(now) {
          const elapsed = now - startTime;
          const progress = Math.min(elapsed / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          const current = Math.floor(target * eased);
          el.textContent = prefix + current.toLocaleString() + suffix;
          if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
      });
      datasetStatObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.3 });
document.querySelectorAll('.dataset-cards').forEach(el => datasetStatObserver.observe(el));

// ---------- Table row hover highlight ----------
document.querySelectorAll('.result-table tbody tr, .comparison-table tbody tr').forEach(row => {
  row.addEventListener('mouseenter', () => {
    row.style.transition = 'background 0.2s';
    if (!row.classList.contains('ours-row') && !row.classList.contains('highlight-row')) {
      row.style.background = 'var(--primary-light)';
    }
  });
  row.addEventListener('mouseleave', () => {
    if (!row.classList.contains('ours-row') && !row.classList.contains('highlight-row')) {
      row.style.background = '';
    }
  });
});

// ---------- Belief card pulse on scroll ----------
const beliefObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const cards = entry.target.querySelectorAll('.belief-card');
      cards.forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        setTimeout(() => {
          card.style.transition = 'all 0.5s cubic-bezier(0.4,0,0.2,1)';
          card.style.opacity = '1';
          card.style.transform = 'translateY(0)';
        }, i * 120);
      });
      beliefObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.2 });
document.querySelectorAll('.belief-layers').forEach(el => beliefObserver.observe(el));

// ---------- Agent card stagger animation ----------
const agentObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const cards = entry.target.querySelectorAll('.agent-card');
      cards.forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'scale(0.9) translateY(16px)';
        setTimeout(() => {
          card.style.transition = 'all 0.5s cubic-bezier(0.4,0,0.2,1)';
          card.style.opacity = '1';
          card.style.transform = 'scale(1) translateY(0)';
        }, i * 100);
      });
      agentObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.2 });
document.querySelectorAll('.agent-grid').forEach(el => agentObserver.observe(el));

// ---------- Coming Soon Modal ----------
function openComingSoon() {
  const modal = document.getElementById('comingSoonModal');
  modal.classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeComingSoon(e) {
  if (e && e.target && !e.target.closest('.modal-box') && e.target !== document.querySelector('.modal-box .btn')) {
    // clicked overlay
  } else if (e && e.target && e.target.closest('.modal-box') && !e.target.closest('.btn')) {
    return; // clicked inside box but not button
  }
  document.getElementById('comingSoonModal').classList.remove('open');
  document.body.style.overflow = '';
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    const modal = document.getElementById('comingSoonModal');
    if (modal.classList.contains('open')) closeComingSoon();
  }
});

// ---------- Hero Parallax on Scroll ----------
(function() {
  const hero = document.getElementById('hero');
  const logoWrap = document.querySelector('.hero-logo-wrap');
  const heroTitle = document.querySelector('.hero-title');

  window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;
    const heroH = hero.offsetHeight;
    if (scrollY > heroH) return;
    const ratio = scrollY / heroH;
    if (logoWrap) logoWrap.style.transform = `translateY(${scrollY * 0.15}px) scale(${1 - ratio * 0.08})`;
    if (heroTitle) heroTitle.style.transform = `translateY(${scrollY * 0.08}px)`;
    hero.style.opacity = 1 - ratio * 0.4;
  }, { passive: true });
})();

// ---------- Hero Quote Typing Effect ----------
(function() {
  const quoteEl = document.querySelector('.hero-quote');
  if (!quoteEl) return;
  const fullText = quoteEl.textContent;
  quoteEl.textContent = '';
  quoteEl.style.borderRight = '2px solid var(--primary)';

  let i = 0;
  function type() {
    if (i < fullText.length) {
      quoteEl.textContent += fullText.charAt(i);
      i++;
      setTimeout(type, 40 + Math.random() * 30);
    } else {
      // Remove cursor after typing done
      setTimeout(() => { quoteEl.style.borderRight = 'none'; }, 1500);
    }
  }
  // Start typing after a delay
  setTimeout(type, 1200);
})();

// ---------- Case Study Card Entrance ----------
const caseObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const cards = entry.target.querySelectorAll('.case-card');
      cards.forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        setTimeout(() => {
          card.style.transition = 'all 0.6s cubic-bezier(0.4,0,0.2,1)';
          card.style.opacity = '1';
          card.style.transform = 'translateY(0)';
        }, i * 200);
      });
      // Animate strategy bars
      const bars = entry.target.querySelectorAll('.bar-fill');
      bars.forEach(bar => {
        const targetWidth = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
          bar.style.width = targetWidth;
        }, 800);
      });
      caseObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.15 });
document.querySelectorAll('.case-grid').forEach(el => caseObserver.observe(el));

// ---------- Highlight-box hover glow ----------
document.querySelectorAll('.highlight-box').forEach(box => {
  box.addEventListener('mouseenter', () => {
    box.style.transition = 'box-shadow 0.3s, transform 0.3s';
    box.style.boxShadow = '0 4px 20px rgba(26,86,219,0.1)';
    box.style.transform = 'translateX(4px)';
  });
  box.addEventListener('mouseleave', () => {
    box.style.boxShadow = '';
    box.style.transform = '';
  });
});

// ---------- Finding-tag click ripple ----------
document.querySelectorAll('.finding-tag').forEach(tag => {
  tag.style.cursor = 'default';
  tag.addEventListener('click', function(e) {
    const ripple = document.createElement('span');
    ripple.style.cssText = 'position:absolute;border-radius:50%;background:rgba(0,0,0,0.1);transform:scale(0);animation:tagRipple 0.5s ease-out;pointer-events:none;';
    const rect = this.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height) * 2;
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
    ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
    this.style.position = 'relative';
    this.style.overflow = 'hidden';
    this.appendChild(ripple);
    setTimeout(() => ripple.remove(), 500);
  });
});

// ---------- Section header label slide-in ----------
const labelObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const label = entry.target.querySelector('.section-label');
      if (label) {
        label.style.opacity = '0';
        label.style.transform = 'translateY(10px) scale(0.8)';
        setTimeout(() => {
          label.style.transition = 'all 0.5s cubic-bezier(0.4,0,0.2,1)';
          label.style.opacity = '1';
          label.style.transform = 'translateY(0) scale(1)';
        }, 100);
      }
      labelObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.3 });
document.querySelectorAll('.section-header').forEach(el => labelObserver.observe(el));

// ---------- KaTeX auto-render on load ----------
document.addEventListener('DOMContentLoaded', () => {
  const checkKatex = setInterval(() => {
    if (window.renderMathInElement) {
      clearInterval(checkKatex);
      renderMathInElement(document.body, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false }
        ],
        throwOnError: false
      });
    }
  }, 100);

  // Reveal already-visible elements
  setTimeout(() => {
    document.querySelectorAll('.reveal').forEach(el => {
      const rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight) el.classList.add('visible');
    });
  }, 200);
});

// ---------- Inject tag ripple keyframe ----------
(function() {
  const style = document.createElement('style');
  style.textContent = '@keyframes tagRipple { to { transform: scale(1); opacity: 0; } }';
  document.head.appendChild(style);
})();

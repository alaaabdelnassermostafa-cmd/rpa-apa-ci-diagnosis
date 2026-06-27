document.addEventListener('DOMContentLoaded', () => {
    const failuresContainer = document.getElementById('failures-container');
    const template = document.getElementById('failure-card-template');

    async function fetchFailures() {
        try {
            const response = await fetch('/api/failures');
            const data = await response.json();
            renderFailures(data);
        } catch (error) {
            console.error('Error fetching failures:', error);
            if (failuresContainer.children.length === 1 && failuresContainer.firstElementChild.classList.contains('loading-state')) {
                failuresContainer.innerHTML = '<div class="loading-state"><p>Error connecting to backend.</p></div>';
            }
        }
    }

    function renderFailures(failures) {
        if (!failures || failures.length === 0) {
            failuresContainer.innerHTML = `
                <div class="glass-card" style="padding: 3rem; text-align: center; color: var(--text-secondary);">
                    <svg style="width: 48px; height: 48px; margin-bottom: 1rem; opacity: 0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
                    <p>No CI/CD failures intercepted yet.</p>
                    <p style="font-size: 0.85rem; margin-top: 0.5rem">Waiting for GitHub webhooks...</p>
                </div>
            `;
            return;
        }

        failuresContainer.innerHTML = '';
        
        failures.forEach(f => {
            const clone = template.content.cloneNode(true);
            const card = clone.querySelector('.glass-card');
            
            // Populate basic info
            clone.querySelector('.repo-name').textContent = f.repo;
            clone.querySelector('.branch-name').textContent = f.branch;
            clone.querySelector('.commit-sha').textContent = f.commit_sha.substring(0, 7);
            
            // Status badge
            const statusBadge = clone.querySelector('.status-badge');
            statusBadge.textContent = f.status;
            statusBadge.classList.add(f.status.toLowerCase());
            
            // Classification
            const categoryPill = clone.querySelector('.category-pill');
            const cat = f.classification_category || 'UNKNOWN';
            categoryPill.textContent = cat;
            categoryPill.classList.add(`cat-${cat}`);
            
            // Remediation Content (Parse Markdown)
            const mdContent = f.remediation_plan || 'Processing...';
            clone.querySelector('.remediation-content').innerHTML = marked.parse(mdContent);
            
            // Timestamp
            const date = new Date(f.timestamp + 'Z'); // SQLite timestamp is UTC
            clone.querySelector('.timestamp').textContent = date.toLocaleString();
            
            // Setup Toggle Interaction
            const toggle = clone.querySelector('.remediation-toggle');
            const remContainer = clone.querySelector('.remediation-container');
            toggle.addEventListener('click', () => {
                remContainer.classList.toggle('open');
            });
            
            failuresContainer.appendChild(clone);
            
            // Stagger entrance animation
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 50);
        });
    }

    // Initial fetch
    fetchFailures();
    
    // Poll every 5 seconds
    setInterval(fetchFailures, 5000);
});

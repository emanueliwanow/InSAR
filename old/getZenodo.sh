REC=15831598   # or use your record id

# Get the file download URLs (works with old/new API layouts), needs jq
curl -s "https://zenodo.org/api/records/$REC" \
| jq -r '(.files[]?.links.self, .files.entries[]?.links.self) // empty' \
> urls.txt
from src.data_fetcher.fetcher import search_trials

def test_pagination():
    # First page request
    print("Testing pagination integration...")
    first_page = search_trials('breast cancer', max_results=5)
    print("\nFirst page result structure:")
    print(f"- Studies count: {len(first_page.get('studies', []))}")
    print(f"- Has pagination: {'pagination' in first_page}")
    
    # Check for next page token
    next_token = first_page.get('nextPageToken')
    if not next_token:
        print("\nNo nextPageToken found in first page response")
        return
    
    print(f"\nNext page token: {next_token}")
    
    # Second page request
    second_page = search_trials('breast cancer', max_results=5, page_token=next_token)
    print("\nSecond page result structure:")
    print(f"- Studies count: {len(second_page.get('studies', []))}")
    print(f"- Has pagination: {'pagination' in second_page}")
    
    print("\nPagination test completed")

if __name__ == '__main__':
    test_pagination()
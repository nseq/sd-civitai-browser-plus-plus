def init():
    global last_version, current_download, cancel_status, recent_model, json_data, json_info, last_dwn, last_start, last_del, main_folder, previous_search_term, previous_tile_count, previous_inputs, download_fail, sortNewest, contentChange, inputs_changed, isDownloading, pageChange, tile_count, old_download, ver
    cancel_status = None
    recent_model = None
    json_data = None
    json_info = None
    last_dwn = None
    last_start = None
    last_del = None
    main_folder = None
    previous_search_term = None
    previous_tile_count = None
    previous_inputs = None
    last_version = None
    current_download = None
    
    download_fail = False
    sortNewest = False
    contentChange = False
    inputs_changed = False
    isDownloading = False
    pageChange = False
    old_download = False
    tile_count = 15
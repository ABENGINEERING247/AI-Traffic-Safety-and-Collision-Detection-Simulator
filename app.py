# --- C. LIVE WEBCAM FEED (BROWSER & HARDWARE COMPATIBLE) ---
elif source_option == "Live Webcam":
    st.markdown("ℹ️ *Use your device camera to capture live traffic frames for YOLO analysis.*")
    
    # Use Streamlit's built-in browser camera input which works both locally and on Cloud deployment
    camera_image = st.camera_input("Capture Live Traffic Snapshot")

    if camera_image is not None:
        # Convert the uploaded buffer to an OpenCV image
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results = model(cv2_img, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()

        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
        active_objects = len(boxes)
        v_counts = classify_all_entities(boxes, results[0].boxes)
        current_risk = False
        min_dist = float("inf")

        if active_objects >= 2:
            for i in range(active_objects):
                for j in range(i + 1, active_objects):
                    b1, b2 = boxes[i], boxes[j]
                    c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
                    c2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
                    dist = np.linalg.norm(c1 - c2)
                    if dist < min_dist:
                        min_dist = dist
                    if dist < proximity_limit:
                        current_risk = True
                        cv2.line(annotated, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 0, 255), 3)

        warning_count = 0
        if current_risk:
            warning_count = 1
            instruction_text = "Slow down instantly, watch for pedestrians/ambulances, and stay defensive."
            cv2.putText(annotated, "WARNING: HAZARD PROXIMITY DETECTED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            status_alert.error("🚨 Collision risk detected in snapshot!")
            if warning_count == 1:
                show_hazard_popup(active_objects, min_dist)
        else:
            instruction_text = "Normal operation. Maintain regular safe speed limits."
            status_alert.success("🟢 Frame stable.")

        st.session_state.simulation_logs.append({
            "Date & Time": current_datetime,
            "Frame / File": f"Webcam Snapshot",
            "Timestamp (s)": 0.0,
            "Active Objects": active_objects,
            "Cars": v_counts["Cars"],
            "Bikes": v_counts["Bikes"],
            "Buses": v_counts["Buses"],
            "Ambulances": v_counts["Ambulances"],
            "Persons": v_counts["Persons"],
            "Persons with Mask": v_counts["Persons with Mask"],
            "Disabled Persons": v_counts["Disabled Persons"],
            "Blind Persons": v_counts["Blind Persons"],
            "Collision Risk Warning": "YES" if current_risk else "NO",
            "Warden Billu Instructions": instruction_text,
        })

        metric_box_objects.metric("Active Objects Tracked", active_objects)
        metric_box_warnings.metric("Collision Warnings", warning_count)
        update_breakdown_display(v_counts)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        media_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

    render_audit_report_section()

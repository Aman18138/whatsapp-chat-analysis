import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import preprocessor
import helper

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt"])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    # Try a few common encodings WhatsApp exports show up in depending on
    # device/OS before giving up.
    data = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            data = bytes_data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if data is None:
        st.error("Couldn't read this file — it doesn't look like a text file.")
        st.stop()

    try:
        with st.spinner("Reading and parsing chat..."):
            df = preprocessor.preprocess(data)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # fetch unique users
    user_list = df["user"].dropna().unique().tolist()
    if "group_notification" in user_list:
        user_list.remove("group_notification")
    user_list.sort()
    is_group_chat = len(user_list) > 2
    user_list.insert(0, "Overall")

    with st.sidebar.expander(f"Chat overview ({df.shape[0]} messages)"):
        st.write(f"**Participants:** {len(user_list) - 1}")
        st.write(f"**From:** {df['only_date'].min()}")
        st.write(f"**To:** {df['only_date'].max()}")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    with st.expander("Show raw parsed data"):
        st.dataframe(df)
        st.download_button(
            "Download parsed data as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="parsed_chat.csv",
            mime="text/csv",
        )

    if st.sidebar.button("Show Analysis"):
        with st.spinner("Crunching numbers..."):
            # Stats Area
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(
                selected_user, df
            )
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        if timeline.empty:
            st.info("No messages to show for this selection.")
        else:
            fig, ax = plt.subplots()
            ax.plot(timeline["time"], timeline["message"], color="green")
            ax.set_ylabel("Messages")
            plt.xticks(rotation="vertical")
            st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        if daily_timeline.empty:
            st.info("No messages to show for this selection.")
        else:
            fig, ax = plt.subplots()
            ax.plot(daily_timeline["only_date"], daily_timeline["message"], color="black")
            ax.set_ylabel("Messages")
            plt.xticks(rotation="vertical")
            st.pyplot(fig)

        # activity map
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color="purple")
            ax.set_ylabel("Messages")
            plt.xticks(rotation="vertical")
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color="orange")
            ax.set_ylabel("Messages")
            plt.xticks(rotation="vertical")
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        if user_heatmap.empty:
            st.info("Not enough data to build a heatmap for this selection.")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(user_heatmap, ax=ax)
            st.pyplot(fig)

        # finding the busiest users in the group (only meaningful for real
        # group chats — for a 1-on-1 conversation this is just two names, so
        # it's skipped to keep the report relevant either way).
        if selected_user == "Overall" and is_group_chat:
            st.title("Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color="red")
                ax.set_ylabel("Messages")
                plt.xticks(rotation="vertical")
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        if df_wc is None:
            st.info("Not enough text (after removing media/stopwords) to build a wordcloud.")
        else:
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            ax.axis("off")
            st.pyplot(fig)

        # most common words
        st.title("Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        if most_common_df.empty:
            st.info("No common words to show for this selection.")
        else:
            fig, ax = plt.subplots()
            ax.barh(most_common_df["word"], most_common_df["count"])
            ax.invert_yaxis()
            st.pyplot(fig)

        # emoji analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        if emoji_df.empty:
            st.info("No emojis found for this selection.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                top_emoji = emoji_df.head()
                ax.pie(top_emoji["count"], labels=top_emoji["emoji"], autopct="%0.2f")
                st.pyplot(fig)

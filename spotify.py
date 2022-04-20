
from argparse import ArgumentError
import sys, getopt
import numpy
import spotipy
import spotipy.util as util 
import requests
import random  
import pandas as pd 
import os 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

username = ""


def auth(createPlayList=False, playlist_name="My Program Recommendations"):
    """
    Authenticates user with the Spotipy API
    """
    global username 
    clientId = "YOUR ID"
    clientSecret = "YOUR SECRET"
    username = "YOUR USERNAME"


    #note that I extended the scope to also modify non-public playlists
    scope = "playlist-modify-public playlist-modify-private playlist-read-private playlist-read-collaborative user-library-read user-read-recently-played user-read-currently-playing user-follow-read  user-read-private user-read-email user-follow-modify"

    client_credentials_manager = spotipy.SpotifyClientCredentials(client_id=clientId, 
                                                        client_secret=clientSecret)

    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

    token = util.prompt_for_user_token(username=username, scope=scope, client_id=clientId, client_secret=clientSecret, redirect_uri="http://localhost:8888/callback")

    sp = spotipy.Spotify(auth=token)

    if createPlayList: 
        id, found = GetPlaylistID(username, playlist_name, sp)
        if not found:   
            sp.user_playlist_create(username, name=playlist_name)
            print("Created")

    return sp 

def GetPlaylistID(username, playlist_name, sp):
    playlist_id = ''
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:  # iterate through playlists I follow
        if playlist['name'] == playlist_name:  # filter for newly created playlist
            playlist_id = playlist['id']
            return playlist_id, True 
    return None, False 

def create_df_playlist(api_results,sp = None, append_audio = True):
    """
    Reads in the spotipy query results for a playlist and returns a 
    DataFrame with track_name,track_id,artist,album,duration,popularity
    and audio_features unless specified otherwise.
    Parameters
    ----------
    api_results : the results of a query to spotify with .recommendations()
    sp : spotfiy authentication token (result of authenticate())
    append_audio : argument to choose whether to append audio features
    Returns
    -------
    df: DataFrame containing track_name, track_id, artist, album, duration, popularity
    """
    #df = create_df_saved_songs(api_results["tracks"])
    df = api_results
    if append_audio == True:
        assert sp != None, "sp needs to be specified for appending audio features"
        df = append_audio_features(df,sp)
    return df
    
def append_audio_features(df,spotify_auth, return_feat_df = False):
    """ 
    Fetches the audio features for all songs in a DataFrame and
    appends these as rows to the DataFrame.
    Requires spotipy to be set up with an auth token.
    Parameters
    ----------
    df : Dataframe containing at least track_name and track_id for spotify songs
    spotify_auth: spotfiy authentication token (result of authenticate())
    return_feat_df: argument to choose whether to also return df with just the audio features
    
    Returns
    -------
    df: DataFrame containing all original rows and audio features for each song
    df_features(optional): DataFrame containing just the audio features
    """
    print("\nAppending Audio Features\n")
    audio_features = spotify_auth.audio_features(df["track_id"][:])
    assert len(audio_features) == len(df["track_id"][:])
    feature_cols = list(audio_features[0].keys())[:-7]
    features_list = []
    for features in audio_features:
        try:
            song_features = [features[col] for col in feature_cols]
            features_list.append(song_features)
        except TypeError:
            pass
    df_features = pd.DataFrame(features_list,columns = feature_cols)
    df = pd.concat([df,df_features],axis = 1)
    if return_feat_df == False:
        return df
    else:
        return df,df_features

def create_df_saved_songs(api_results):
    """
    Reads in the spotipy query results for user saved songs and returns a DataFrame with
    track_name,track_id, artist,album,duration,popularity
    Parameters
    ----------
    api_results : the results of a query to spotify with .current_user_saved_tracks()
    Returns
    -------
    df: DataFrame containing track_name,track_id, artist,album,duration,popularity
    """
    #create lists for df-columns
    track_name = []
    track_id = []
    artist = []
    album = []
    duration = []
    popularity = []
    print(type(api_results["items"]))
    #loop through api_results
    for items in api_results["items"]:
        try:
            track_name.append(items["track"]['name'])
            track_id.append(items["track"]['id'])
            artist.append(items["track"]["artists"][0]["name"])
            duration.append(items["track"]["duration_ms"])
            album.append(items["track"]["album"]["name"])
            popularity.append(items["track"]["popularity"])
        except TypeError: 
            pass
    # Create the final df   
    df = pd.DataFrame({ "track_name": track_name, 
                             "album": album, 
                             "track_id": track_id,
                             "artist": artist, 
                             "duration": duration, 
                             "popularity": popularity})
    return df

def playlist_df(playlist_uri, sp):
    #get the playlist data from the API
    playlist = sp.playlist(playlist_uri)
    playlist_df = create_df_playlist(playlist,sp = sp)

    #get seed tracks for recommendations
    seed_tracks = playlist_df["track_id"].tolist()

    #create recommendation df from multiple recommendations
    recomm_dfs = []
    for i in range(5,len(seed_tracks)+1,5):
        recomms = sp.recommendations(seed_tracks = seed_tracks[i-5:i],limit = 25)
        recomms_df = append_audio_features(create_df_recommendations(recomms),sp)
        recomm_dfs.append(recomms_df)
    recomms_df = pd.concat(recomm_dfs)
    recomms_df.reset_index(drop = True, inplace = True)

def create_df_recommendations(api_results):
    """
    Reads in the spotipy query results for spotify recommended songs and returns a 
    DataFrame with track_name,track_id,artist,album,duration,popularity
    Parameters
    ----------
    api_results : the results of a query to spotify with .recommendations()
    Returns
    -------
    df: DataFrame containing track_name, track_id, artist, album, duration, popularity
    """
    track_name = []
    track_id = []
    artist = []
    album = []
    duration = []
    popularity = []
    for items in api_results['tracks']:
        try:
            track_name.append(items['name'])
            track_id.append(items['id'])
            artist.append(items["artists"][0]["name"])
            duration.append(items["duration_ms"])
            album.append(items["album"]["name"])
            popularity.append(items["popularity"])
        except TypeError:
            pass
        df = pd.DataFrame({ "track_name": track_name, 
                                "album": album, 
                                "track_id": track_id,
                                "artist": artist, 
                                "duration": duration, 
                                "popularity": popularity})

    return df

def create_similarity_score(df1,df2,similarity_score = "cosine_sim"):
    """ 
    Creates a similarity matrix for the audio features (except key and mode) of two Dataframes.
    Parameters
    ----------
    df1 : DataFrame containing track_name,track_id, artist,album,duration,popularity
            and all audio features
    df2 : DataFrame containing track_name,track_id, artist,album,duration,popularity
            and all audio features
    
    similarity_score: similarity measure (linear,cosine_sim)
    Returns
    -------
    A matrix of similarity scores for the audio features of both DataFrames.
    """
    
    assert list(df1.columns[6:]) == list(df2.columns[6:]), "dataframes need to contain the same columns"
    features = list(df1.columns[6:])
    features.remove('key')
    features.remove('mode')
    df_features1,df_features2 = df1[features],df2[features]
    scaler = MinMaxScaler() #StandardScaler() not used anymore
    df_features_scaled1,df_features_scaled2 = scaler.fit_transform(df_features1),scaler.fit_transform(df_features2)
    if similarity_score == "linear":
        linear_sim = linear_kernel(df_features_scaled1, df_features_scaled2)
        return linear_sim
    elif similarity_score == "cosine_sim":
        cosine_sim = cosine_similarity(df_features_scaled1, df_features_scaled2)
        return cosine_sim
    #other measures may be implemented in the future

def main(argv):
    playlist_name = f"My Program Recommendations"
    for opt in range(0, len(argv)):
        if argv[opt] == "-p":
            playlist_name = argv[opt + 1]
            break 
    print(str(playlist_name))
    sp = auth(True, playlist_name)
    limit = 30 
    trackId = []
    playList = sp.current_user_saved_tracks(limit=limit).get('items')
    #Offset is 7
    for i in range(0, limit):
        trackId.append(playList[i].get('track').get('id'))
        print(str(playList[i].get('track')))
    #print(str(trackId[0]))
    chosenTracks = []
    #Randomize the position 
    start = random.randrange(0, limit - 6)
    for i in range(start, start + 5):
        chosenTracks.append(trackId[i]) 

    Min = False 
    Max = True
    value = "" 
    #opts, args = getopt.getopt(argv,"t:-:+:e:v:")
    print(str(argv))
    for opt in argv:
        if opt in ['-min']:
            Min = True
            Max = False
        elif opt in ['-help']:
            print("+ for max; - for min; p for playlist selection other than standard; e for energy; v for valence; t for tempo")
            os.abort() 
        elif opt in ['-max']:
            Max = True 
            Min = False 
        elif opt in ['-t']:
            value = "tempo" 
        elif opt in ['-e']:
            value = "energy"
        elif opt in ['-v']:
            value = "valence"     
        elif opt in ['-live']:
            value = "liveness"
        elif opt in ['-loud']:
            value = "loudness"
        elif opt in ['-i']:
            value = "instrumentalness"
        elif opt in ['-a']:
            value = "acousticness"    
        elif opt in ['-random']:
            Min = False 
            Max = False             

    print("Value: " + value)
    print("Min: " + str(Min))
    print("Max: " + str(Max))
    useRec = True 
    numberOfTracks = 5 
    savedSongs = sp.current_user_saved_tracks(limit=limit) 
    #ADD FLAG TO CHOOSE SEARCH BETWEEN PLAYLIST, ALBUM, OR ARTIST 
    for opt in range(0, len(argv)):
        if argv[opt] == '-search':
            chosenTracks = []
            listDict = []
            searched = sp.search(q=str(argv[opt + 1]), type="playlist", limit=2)
            for i in range(0,len(searched.get('playlists').get('items'))):
                id = (str(searched.get('playlists').get('items')[i].get('id')))
                tracks = sp.playlist_tracks(playlist_id=id)
                for j in range(0, len(tracks.get('items'))):
                    chosenTracks.append(str(tracks.get('items')[j].get('track').get('id')))
                print(str(len(chosenTracks)))
                dic = sp.tracks(tracks=chosenTracks[0:49])
                print(str(type(dic)))
                listDict.append(dic)
                chosenTracks = []
            useRec = False 
        elif argv[opt] == '-num':
            if int(argv[opt + 1]) <= 0:
                raise ValueError("Number of tracks chosen cannot be less than 1") 
            else:
                numberOfTracks = int(argv[opt + 1])
                
    #ALLOW TRACKS FROM SEARCHING TO BE SORTED AND CHOSEN BASED OFF OF CHARACTERISTICS; SEARCH WOULD NOT USE RECOMMENDATIONS  
    #APPEND AUDIO FEATURES AND THEN CHOOSE HIGHEST 
    saveDf = create_df_saved_songs(api_results=savedSongs)
    saveDf = append_audio_features(df=saveDf, spotify_auth=sp)

    if useRec:
        rec = sp.recommendations(seed_tracks=chosenTracks, limit=30)
        df = create_df_recommendations(api_results=rec)
        df2 = create_df_playlist(api_results=df, sp=sp)
    else:
        tempFrame = pd.DataFrame()
        for i in range(0, len(listDict)):
            print(str(listDict[i]))
            df = create_df_recommendations(api_results=listDict[i])
            df2 = append_audio_features(df=df, spotify_auth=sp)
            print(str(df2))
            data = [df2, tempFrame]
            tempFrame = pd.concat(data, ignore_index=True)
        print(str(tempFrame))
        df2 = tempFrame
        print(str(df2))
    print(str(df2))
    score = None 
    weightScore = []
    weight = 0 

    #Another functionality: allow users to specify a playlist to base recommendations 
    if value == "":
        score = create_similarity_score(df1=saveDf, df2=df2)
        weightScore = []
        weight = 0
        for i in range(0, len(score)): 
            for j in range(0, len(score[i])):
                weight += score[i][j]
            weightScore.append(weight / len(score[i]))
            weight = 0

    else:
        print(str(df2.keys()))
        score = df2[value] 
        print(df2)
        #print(str(score))
        weightScore = []
        weight = 0
        print(str(len(score)))
        for i in range(0, len(score)): 
            weight += score[i]
            weightScore.append(weight)
            weight = 0

    finalDict = {}
    for i in range(0, len(weightScore)):
        finalDict.update({str(df2['track_id'][i]) : weightScore[i]})

    finalTrackId = []
    
    if Max:
        for i in range(0, numberOfTracks):
            trackId = max(finalDict, key=finalDict.__getitem__)
            print(str(sp.track(track_id=trackId)))
            finalTrackId.append(trackId)
            finalDict.pop(str(trackId))
    elif Min:
        for i in range(0, numberOfTracks):
            trackId = min(finalDict, key=finalDict.__getitem__)
            print(str(sp.track(track_id=trackId)))
            finalTrackId.append(trackId)
            finalDict.pop(str(trackId))
    else:
        for i in range(0, numberOfTracks):
            trackId = random.choice(list(finalDict))
            print(str(trackId))
            print(str(sp.track(track_id=trackId)))
            finalTrackId.append(trackId)
            finalDict.pop(str(trackId))        

    playId, found = GetPlaylistID(username=username, playlist_name=playlist_name, sp=sp)

    sp.user_playlist_add_tracks(user=username, playlist_id = playId, tracks=finalTrackId)
    print("Finished")

if __name__ == "__main__":
   main(sys.argv[1:])




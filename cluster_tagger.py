import relevancer as rlv


def tag_clusters(input_collection, tagged_collection):

	i = 0

	for cluster in input_collection.find():
    
		i += 1
    
		print("\n\nSome tweets in the cluster " + str(i) + " ; \n")
    
		for tweet in cluster["ctweettuplelist"][:5]:
			print(tweet[2])
	   
		a = len(cluster["ctweettuplelist"])
      
		for tweet in cluster["ctweettuplelist"][a-5:a]:
			print(tweet[2])

		objID = cluster["_id"]
    
		if (tagged_collection.find({"_id" : objID}).count() > 0):

			print("\nWarning : These tweets are already tagged as \"" 
				+ tagged_collection.find_one({"_id" : objID})["class"] + "\".")

			while True:
				check_it = (input("Are they tagged correctly? [Y/n]: ") or "y").lower()
				if check_it not in ["y","yes", "n", "no"]:
					print("\nError : Argument is invalid. Please enter again [Y/n]:")
				else:
					break

			if (check_it in ["y","yes"]):

				continue;

			elif (check_it in ["n", "no"]):

				tagged_collection.remove({"_id" : objID})
	            
				modified = input("The corret tag of these tweets is... (test, futbol, politics) : ")
	          
				cluster["class"] = modified
	            
				tagged_collection.insert(cluster)
		    
	            
		else:
	        
			tag = input("These tweets are about... (test, futbol, politics) : ")
	    
			cluster["class"] = tag
	    
			tagged_collection.insert(cluster)
	

	print("All clusters are tagged.")



if __name__ == "__main__":

	myalldata, all_data_clusters = rlv.connect_mongodb(configfile='myalldata.ini', coll_name='all_data_clusters')

	myalldata2, tagged_clusters = rlv.connect_mongodb(configfile='myalldata.ini', coll_name='tagged_clusters')

	tag_clusters(all_data_clusters, tagged_clusters)







from gym.envs.registration import register

register(
    id='RLClassification-v0',
    entry_point='EnvRLforClassification.envs.env_4_rl_classification:Env4RLClassification',
    #kwargs={X:None,y:None,batch_size:None,output_shape:None,randomize:False,custom_rewards:None}
)

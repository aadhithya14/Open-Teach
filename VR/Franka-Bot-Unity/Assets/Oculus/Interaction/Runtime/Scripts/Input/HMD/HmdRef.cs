/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System;
using UnityEngine;
using UnityEngine.Assertions;

namespace Oculus.Interaction.Input
{
    /// <summary>
    /// A set of constants that are passed to each child of a Hand modifier tree from the root DataSource.
    /// </summary>
    public class HmdRef : MonoBehaviour, IHmd
    {
        [SerializeField, Interface(typeof(Hmd))]
        private MonoBehaviour _hmd;
        private IHmd Hmd;

        public event Action HmdUpdated
        {
            add => Hmd.HmdUpdated += value;
            remove => Hmd.HmdUpdated -= value;
        }

        protected virtual void Awake()
        {
            Hmd = _hmd as IHmd;
        }

        protected virtual void Start()
        {
            Assert.IsNotNull(Hmd);
        }

        public bool GetRootPose(out Pose pose)
        {
            return Hmd.GetRootPose(out pose);
        }

        #region Inject
        public void InjectAllHmdRef(IHmd hmd)
        {
            InjectHmd(hmd);
        }

        public void InjectHmd(IHmd hmd)
        {
            _hmd = hmd as MonoBehaviour;
            Hmd = hmd;
        }
        #endregion
    }
}

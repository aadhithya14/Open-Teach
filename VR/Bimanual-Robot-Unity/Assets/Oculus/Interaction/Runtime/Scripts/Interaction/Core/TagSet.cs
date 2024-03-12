/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace Oculus.Interaction
{
    /// <summary>
    /// A Tag Set that can be added to a GameObject with an Interactable to be filtered against.
    /// </summary>
    public class TagSet : MonoBehaviour
    {
        [SerializeField]
        private List<string> _tags;

        private HashSet<string> _tagSet;

        protected virtual void Start()
        {
            _tagSet = new HashSet<string>();
            foreach (string tag in _tags)
            {
                _tagSet.Add(tag);
            }
        }

        public bool ContainsTag(string tag) => _tagSet.Contains(tag);

        public void AddTag(string tag) => _tagSet.Add(tag);
        public void RemoveTag(string tag) => _tagSet.Remove(tag);

        #region Inject

        public void InjectOptionalTags(List<string> tags)
        {
            _tags = tags;
        }

        #endregion
    }
}
